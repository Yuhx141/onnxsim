/*
 * axcl_batch_runner: a persistent AXCL model runner for the LXD guest.
 *
 * axcl_run_model loads the model, pushes inputs and runs once per process, and
 * each invocation through the VM harness costs an `lxc file push` plus a
 * runtime init. A training step is hundreds of small models run in sequence,
 * so this keeps one process (one axclInit, one device context) alive and takes
 * line commands on stdin; tensors travel as raw files on the virtiofs share
 * (/mnt/share), never through stdin/stdout.
 *
 *   LOAD <model path>               -> OK <id>, one "IN|OUT <idx> <name> <bytes>
 *                                      <dtype> <dims...>" line per tensor, END
 *   RUN <id> <n_in> <in files...> <n_out> <out files...>
 *                                   -> OK <execute microseconds>
 *   RUNR <id> <n_in> <in files...> <n_keep> <input output pairs...>
 *        <n_out> <out files...>       -> same, retaining selected state on device
 *   UNLOAD <id>                     -> OK
 *   TPUT <name> <file> <bytes>      -> OK   (host file -> a named device tensor)
 *   TGET <name> <file>              -> OK <bytes>   (device tensor -> host file)
 *   TDEL <name> | TCLEAR            -> OK
 *   RUNT <id> <n_in> <tensors...> <n_out> <tensors...>
 *                                   -> OK <execute microseconds>; inputs come
 *                                      from named device tensors and outputs
 *                                      go to named device tensors ("-" drops
 *                                      one), by device-to-device copy: tensors
 *                                      pass between models without crossing the
 *                                      VM boundary
 *   QUIT
 *
 * Every error answers "ERR <text>" and keeps the process alive. Build in the
 * guest: gcc -O2 -o axrun axcl_batch_runner.c -I/usr/include/axcl
 *        -L/usr/lib/axcl -laxcl_rt -laxcl_sys -Wl,-rpath,/usr/lib/axcl
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "axcl.h"

#define MAX_MODELS 4096
#define MAX_IO 64

typedef struct {
  int used;
  uint64_t model_id, ctx_id;
  axclrtEngineIOInfo info;
  axclrtEngineIO io;
  uint32_t n_in, n_out;
  void *in_dev[MAX_IO], *out_dev[MAX_IO];
  uint64_t in_size[MAX_IO], out_size[MAX_IO];
  int keep_in[MAX_IO];
} Model;

static Model models[MAX_MODELS];
static void *host_buf;
static uint64_t host_cap;

static void *host(uint64_t n) {
  if (n > host_cap) {
    free(host_buf);
    host_buf = malloc(n);
    host_cap = host_buf ? n : 0;
  }
  return host_buf;
}

static void unload(Model *m) {
  for (uint32_t i = 0; i < m->n_in; i++) axclrtFree(m->in_dev[i]);
  for (uint32_t i = 0; i < m->n_out; i++) axclrtFree(m->out_dev[i]);
  axclrtEngineDestroyIO(m->io);
  axclrtEngineDestroyIOInfo(m->info);
  axclrtEngineUnload(m->model_id);
  memset(m, 0, sizeof(*m));
}

static void print_dims(axclrtEngineIODims *d) {
  for (int k = 0; k < d->dimCount; k++) printf(" %d", d->dims[k]);
}

static void cmd_load(const char *path) {
  int slot = -1;
  for (int i = 0; i < MAX_MODELS; i++)
    if (!models[i].used) { slot = i; break; }
  if (slot < 0) { printf("ERR model table full\n"); return; }
  Model *m = &models[slot];
  memset(m, 0, sizeof(*m));
  axclError e;
  if ((e = axclrtEngineLoadFromFile(path, &m->model_id))) {
    printf("ERR load %s: 0x%x\n", path, e); return;
  }
  if ((e = axclrtEngineCreateContext(m->model_id, &m->ctx_id)) ||
      (e = axclrtEngineGetIOInfo(m->model_id, &m->info)) ||
      (e = axclrtEngineCreateIO(m->info, &m->io))) {
    axclrtEngineUnload(m->model_id);
    printf("ERR setup %s: 0x%x\n", path, e); return;
  }
  m->used = 1;
  m->n_in = axclrtEngineGetNumInputs(m->info);
  m->n_out = axclrtEngineGetNumOutputs(m->info);
  if (m->n_in > MAX_IO || m->n_out > MAX_IO) {
    printf("ERR too many io\n"); unload(m); return;
  }
  for (uint32_t i = 0; i < m->n_in; i++) {
    m->in_size[i] = axclrtEngineGetInputSizeByIndex(m->info, 0, i);
    if ((e = axclrtMalloc(&m->in_dev[i], m->in_size[i], AXCL_MEM_MALLOC_NORMAL_ONLY)) ||
        (e = axclrtEngineSetInputBufferByIndex(m->io, i, m->in_dev[i], m->in_size[i]))) {
      m->n_in = i; printf("ERR in alloc 0x%x\n", e); unload(m); return;
    }
  }
  for (uint32_t i = 0; i < m->n_out; i++) {
    m->out_size[i] = axclrtEngineGetOutputSizeByIndex(m->info, 0, i);
    if ((e = axclrtMalloc(&m->out_dev[i], m->out_size[i], AXCL_MEM_MALLOC_NORMAL_ONLY)) ||
        (e = axclrtEngineSetOutputBufferByIndex(m->io, i, m->out_dev[i], m->out_size[i]))) {
      m->n_out = i; printf("ERR out alloc 0x%x\n", e); unload(m); return;
    }
  }
  printf("OK %d\n", slot);
  for (uint32_t i = 0; i < m->n_in; i++) {
    axclrtEngineIODims d; axclrtEngineDataType t = 0;
    axclrtEngineGetInputDims(m->info, 0, i, &d);
    axclrtEngineGetInputDataType(m->info, i, &t);
    printf("IN %u %s %lu %d", i, axclrtEngineGetInputNameByIndex(m->info, i),
           (unsigned long)m->in_size[i], (int)t);
    print_dims(&d); printf("\n");
  }
  for (uint32_t i = 0; i < m->n_out; i++) {
    axclrtEngineIODims d; axclrtEngineDataType t = 0;
    axclrtEngineGetOutputDims(m->info, 0, i, &d);
    axclrtEngineGetOutputDataType(m->info, i, &t);
    printf("OUT %u %s %lu %d", i, axclrtEngineGetOutputNameByIndex(m->info, i),
           (unsigned long)m->out_size[i], (int)t);
    print_dims(&d); printf("\n");
  }
  printf("END\n");
}

static int read_file(const char *p, void *dst, uint64_t n) {
  FILE *f = fopen(p, "rb");
  if (!f) return -1;
  uint64_t got = fread(dst, 1, n, f);
  fclose(f);
  return got == n ? 0 : -2;
}

static int write_file(const char *p, const void *src, uint64_t n) {
  FILE *f = fopen(p, "wb");
  if (!f) return -1;
  uint64_t put = fwrite(src, 1, n, f);
  fclose(f);
  return put == n ? 0 : -2;
}

static void cmd_run(char *args, int resident) {
  char *save = NULL;
  char *tok = strtok_r(args, " ", &save);
  int id = tok ? atoi(tok) : -1;
  if (id < 0 || id >= MAX_MODELS || !models[id].used) { printf("ERR bad id\n"); return; }
  Model *m = &models[id];
  if (!resident) memset(m->keep_in, 0, sizeof(m->keep_in));
  tok = strtok_r(NULL, " ", &save);
  uint32_t n_in = tok ? (uint32_t)atoi(tok) : 0;
  if (n_in != m->n_in) { printf("ERR model has %u inputs\n", m->n_in); return; }
  for (uint32_t i = 0; i < n_in; i++) {
    char *p = strtok_r(NULL, " ", &save);
    void *h = m->keep_in[i] ? NULL : host(m->in_size[i]);
    if (!p || (!m->keep_in[i] && (!h || read_file(p, h, m->in_size[i])))) {
      printf("ERR read input %u (%s, %lu bytes)\n", i, p ? p : "-", (unsigned long)m->in_size[i]);
      return;
    }
    if (!m->keep_in[i]) {
      axclError e = axclrtMemcpy(m->in_dev[i], h, m->in_size[i], AXCL_MEMCPY_HOST_TO_DEVICE);
      if (e) { printf("ERR h2d 0x%x\n", e); return; }
    }
  }
  uint32_t n_keep = 0, keep_in[MAX_IO], keep_out[MAX_IO];
  if (resident) {
    tok = strtok_r(NULL, " ", &save);
    n_keep = tok ? (uint32_t)atoi(tok) : 0;
    if (n_keep > MAX_IO) { printf("ERR too many resident pairs\n"); return; }
    for (uint32_t i = 0; i < n_keep; i++) {
      char *in = strtok_r(NULL, " ", &save);
      char *out = strtok_r(NULL, " ", &save);
      keep_in[i] = in ? (uint32_t)atoi(in) : MAX_IO;
      keep_out[i] = out ? (uint32_t)atoi(out) : MAX_IO;
      if (keep_in[i] >= m->n_in || keep_out[i] >= m->n_out) {
        printf("ERR bad resident pair\n"); return;
      }
    }
  }
  tok = strtok_r(NULL, " ", &save);
  uint32_t n_out = tok ? (uint32_t)atoi(tok) : 0;
  if (n_out != m->n_out) { printf("ERR model has %u outputs\n", m->n_out); return; }
  char *out_path[MAX_IO];
  for (uint32_t i = 0; i < n_out; i++) {
    out_path[i] = strtok_r(NULL, " ", &save);
    if (!out_path[i]) { printf("ERR output path %u\n", i); return; }
  }
  struct timeval t0, t1;
  gettimeofday(&t0, NULL);
  axclError e = axclrtEngineExecute(m->model_id, m->ctx_id, 0, m->io);
  gettimeofday(&t1, NULL);
  if (e) { printf("ERR execute 0x%x\n", e); return; }
  for (uint32_t i = 0; i < n_out; i++) {
    void *h = host(m->out_size[i]);
    if (!h) { printf("ERR output path %u\n", i); return; }
    e = axclrtMemcpy(h, m->out_dev[i], m->out_size[i], AXCL_MEMCPY_DEVICE_TO_HOST);
    if (e) { printf("ERR d2h 0x%x\n", e); return; }
    if (write_file(out_path[i], h, m->out_size[i])) { printf("ERR write %s\n", out_path[i]); return; }
  }
  if (resident) {
    for (uint32_t i = 0; i < n_keep; i++) {
      e = axclrtMemcpy(m->in_dev[keep_in[i]], m->out_dev[keep_out[i]],
                       m->in_size[keep_in[i]], AXCL_MEMCPY_DEVICE_TO_DEVICE);
      if (e) { printf("ERR d2d 0x%x\n", e); return; }
      m->keep_in[keep_in[i]] = 1;
    }
  }
  printf("OK %ld\n", (long)((t1.tv_sec - t0.tv_sec) * 1000000L + (t1.tv_usec - t0.tv_usec)));
}

#define MAX_TENSORS 8192
typedef struct {
  int used;
  char name[128];
  void *dev;
  uint64_t size;
} Tensor;
static Tensor tensors[MAX_TENSORS];

static Tensor *tensor_find(const char *name) {
  for (int i = 0; i < MAX_TENSORS; i++)
    if (tensors[i].used && !strcmp(tensors[i].name, name)) return &tensors[i];
  return NULL;
}

/* The tensor `name` with room for `size` bytes (reallocated when the size differs). */
static Tensor *tensor_get(const char *name, uint64_t size) {
  Tensor *t = tensor_find(name);
  if (t && t->size == size) return t;
  if (t) { axclrtFree(t->dev); memset(t, 0, sizeof(*t)); }
  else {
    for (int i = 0; i < MAX_TENSORS; i++)
      if (!tensors[i].used) { t = &tensors[i]; break; }
    if (!t) return NULL;
  }
  if (strlen(name) >= sizeof(t->name)) return NULL;
  if (axclrtMalloc(&t->dev, size, AXCL_MEM_MALLOC_NORMAL_ONLY)) return NULL;
  t->used = 1;
  t->size = size;
  snprintf(t->name, sizeof(t->name), "%s", name);
  return t;
}

static void tensor_free(Tensor *t) {
  if (t && t->used) { axclrtFree(t->dev); memset(t, 0, sizeof(*t)); }
}

static void cmd_tput(char *args) {
  char *save = NULL;
  char *name = strtok_r(args, " ", &save);
  char *path = strtok_r(NULL, " ", &save);
  char *nb = strtok_r(NULL, " ", &save);
  if (!name || !path || !nb) { printf("ERR tput args\n"); return; }
  uint64_t n = strtoull(nb, NULL, 10);
  Tensor *t = tensor_get(name, n);
  void *h = host(n);
  if (!t || !h) { printf("ERR tput alloc %s (%lu bytes)\n", name, (unsigned long)n); return; }
  if (read_file(path, h, n)) { printf("ERR tput read %s\n", path); return; }
  axclError e = axclrtMemcpy(t->dev, h, n, AXCL_MEMCPY_HOST_TO_DEVICE);
  if (e) { printf("ERR tput h2d 0x%x\n", e); return; }
  printf("OK\n");
}

static void cmd_tget(char *args) {
  char *save = NULL;
  char *name = strtok_r(args, " ", &save);
  char *path = strtok_r(NULL, " ", &save);
  Tensor *t = name ? tensor_find(name) : NULL;
  if (!t || !path) { printf("ERR tget %s\n", name ? name : "-"); return; }
  void *h = host(t->size);
  axclError e = h ? axclrtMemcpy(h, t->dev, t->size, AXCL_MEMCPY_DEVICE_TO_HOST) : -1;
  if (e) { printf("ERR tget d2h 0x%x\n", e); return; }
  if (write_file(path, h, t->size)) { printf("ERR tget write %s\n", path); return; }
  printf("OK %lu\n", (unsigned long)t->size);
}

static void cmd_runt(char *args) {
  char *save = NULL;
  char *tok = strtok_r(args, " ", &save);
  int id = tok ? atoi(tok) : -1;
  if (id < 0 || id >= MAX_MODELS || !models[id].used) { printf("ERR bad id\n"); return; }
  Model *m = &models[id];
  memset(m->keep_in, 0, sizeof(m->keep_in));
  tok = strtok_r(NULL, " ", &save);
  uint32_t n_in = tok ? (uint32_t)atoi(tok) : 0;
  if (n_in != m->n_in) { printf("ERR model has %u inputs\n", m->n_in); return; }
  for (uint32_t i = 0; i < n_in; i++) {
    char *name = strtok_r(NULL, " ", &save);
    Tensor *t = name ? tensor_find(name) : NULL;
    if (!t) { printf("ERR no tensor %s\n", name ? name : "-"); return; }
    if (t->size != m->in_size[i]) {
      printf("ERR tensor %s has %lu bytes, input %u wants %lu\n", name,
             (unsigned long)t->size, i, (unsigned long)m->in_size[i]);
      return;
    }
    axclError e = axclrtMemcpy(m->in_dev[i], t->dev, t->size, AXCL_MEMCPY_DEVICE_TO_DEVICE);
    if (e) { printf("ERR d2d in 0x%x\n", e); return; }
  }
  tok = strtok_r(NULL, " ", &save);
  uint32_t n_out = tok ? (uint32_t)atoi(tok) : 0;
  if (n_out != m->n_out) { printf("ERR model has %u outputs\n", m->n_out); return; }
  char *out_name[MAX_IO];
  for (uint32_t i = 0; i < n_out; i++) {
    out_name[i] = strtok_r(NULL, " ", &save);
    if (!out_name[i]) { printf("ERR output name %u\n", i); return; }
  }
  struct timeval t0, t1;
  gettimeofday(&t0, NULL);
  axclError e = axclrtEngineExecute(m->model_id, m->ctx_id, 0, m->io);
  gettimeofday(&t1, NULL);
  if (e) { printf("ERR execute 0x%x\n", e); return; }
  for (uint32_t i = 0; i < n_out; i++) {
    if (!strcmp(out_name[i], "-")) continue;
    Tensor *t = tensor_get(out_name[i], m->out_size[i]);
    if (!t) { printf("ERR out alloc %s\n", out_name[i]); return; }
    e = axclrtMemcpy(t->dev, m->out_dev[i], m->out_size[i], AXCL_MEMCPY_DEVICE_TO_DEVICE);
    if (e) { printf("ERR d2d out 0x%x\n", e); return; }
  }
  printf("OK %ld\n", (long)((t1.tv_sec - t0.tv_sec) * 1000000L + (t1.tv_usec - t0.tv_usec)));
}

int main(void) {
  setvbuf(stdout, NULL, _IOLBF, 0);
  axclError e;
  if ((e = axclInit("/usr/bin/axcl/axcl.json"))) { printf("ERR axclInit 0x%x\n", e); return 1; }
  axclrtDeviceList devs;
  if ((e = axclrtGetDeviceList(&devs)) || devs.num == 0) {
    printf("ERR no device 0x%x\n", e); axclFinalize(); return 1;
  }
  if ((e = axclrtSetDevice(devs.devices[0])) ||
      (e = axclrtEngineInit(AXCL_VNPU_DISABLE))) {
    printf("ERR device init 0x%x\n", e); axclFinalize(); return 1;
  }
  printf("READY %d\n", devs.devices[0]);
  static char line[1 << 16];
  while (fgets(line, sizeof line, stdin)) {
    line[strcspn(line, "\r\n")] = 0;
    if (!strncmp(line, "LOAD ", 5)) cmd_load(line + 5);
    else if (!strncmp(line, "RUNT ", 5)) cmd_runt(line + 5);
    else if (!strncmp(line, "TPUT ", 5)) cmd_tput(line + 5);
    else if (!strncmp(line, "TGET ", 5)) cmd_tget(line + 5);
    else if (!strncmp(line, "TDEL ", 5)) {
      tensor_free(tensor_find(line + 5));
      printf("OK\n");
    } else if (!strcmp(line, "TCLEAR")) {
      for (int i = 0; i < MAX_TENSORS; i++) tensor_free(&tensors[i]);
      printf("OK\n");
    }
    else if (!strncmp(line, "RUNR ", 5)) cmd_run(line + 5, 1);
    else if (!strncmp(line, "RUN ", 4)) cmd_run(line + 4, 0);
    else if (!strncmp(line, "UNLOAD ", 7)) {
      int id = atoi(line + 7);
      if (id >= 0 && id < MAX_MODELS && models[id].used) { unload(&models[id]); printf("OK\n"); }
      else printf("ERR bad id\n");
    } else if (!strcmp(line, "QUIT")) break;
    else printf("ERR unknown command\n");
  }
  for (int i = 0; i < MAX_TENSORS; i++) tensor_free(&tensors[i]);
  for (int i = 0; i < MAX_MODELS; i++)
    if (models[i].used) unload(&models[i]);
  axclrtEngineFinalize();
  axclrtResetDevice(devs.devices[0]);
  axclFinalize();
  return 0;
}
