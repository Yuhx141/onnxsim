#include "node_api.h"

extern "C" {

void* init_dora_context_from_env(void) { return reinterpret_cast<void*>(1); }
void* dora_next_event(void*) { return nullptr; }
void free_dora_event(void*) {}
void free_dora_context(void*) {}
DoraEventType read_dora_event_type(void*) { return DoraEventType_Stop; }
int read_dora_input_id(void*, char**, size_t*) { return 0; }
int read_dora_input_data(void*, char**, size_t*) { return 0; }
int dora_send_output(void*, const char*, size_t, const char*, size_t) { return 0; }
void dora_log(void*, const char*, size_t, const char*, size_t) {}

}
