#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum DoraEventType {
  DoraEventType_Input = 0,
  DoraEventType_Stop = 1,
} DoraEventType;

void* init_dora_context_from_env(void);
void* dora_next_event(void* context);
void free_dora_event(void* event);
void free_dora_context(void* context);
DoraEventType read_dora_event_type(void* event);
int read_dora_input_id(void* event, char** id, size_t* length);
int read_dora_input_data(void* event, char** data, size_t* length);
int dora_send_output(void* context, const char* id, size_t id_length,
                     const char* data, size_t data_length);
void dora_log(void* context, const char* level, size_t level_length,
              const char* message, size_t message_length);

#ifdef __cplusplus
}
#endif
