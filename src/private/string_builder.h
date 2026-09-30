#ifndef STRING_BUILDER_H
#define STRING_BUILDER_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
  size_t len;
  size_t cap;
  char* buf;
} StringBuilder;


// Inits string builder, returns -1 on failure 0 otherwise
int sb_init(StringBuilder* sb) {
  if(!sb)
    return -1;

  sb->len = 1;
  sb->cap = 128;
  sb->buf = malloc(128);

  if(!sb->buf)
    return -1;

  sb->buf[0] = '\0';
  
  return 0;
}

int sb_append(StringBuilder* sb, const char* str) {
  if(!sb || !str)
    return -1;

  const size_t str_len = strlen(str);
  
  if(str_len + sb->len > sb->cap) {
    size_t new_cap = sb->cap;

    while(new_cap < str_len + sb->len) new_cap *= 2;
    
    char* new_buf;
    if(!(new_buf = realloc(sb->buf, new_cap)))
      return -1;
  
    sb->cap = new_cap;
    sb->buf = new_buf;
  }
  
  sb->len += str_len;
  

  if(!strcat(sb->buf, str))
    return -1;

  return 0;
}

#endif
