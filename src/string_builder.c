#include <stddef.h>
#include "string_builder.h"

#include <stdlib.h>
#include <string.h>

void init_sb(StringBuilder* sb) {
    sb->buff = malloc(128);
    sb->cap = 128;
    sb->len = 0;
    sb->buff[0] = '\0';
}

void sb_append(StringBuilder* sb, const char* string) {
    const size_t string_len = strlen(string);
    if (sb->len + string_len < sb->cap) {
        memcpy(&sb->buff[sb->len], string, string_len);
        sb->len += string_len;
        sb->buff[sb->len] = '\0';
    }else {
        size_t new_cap = sb->cap;
        while (sb->len + string_len + 1 > new_cap) {
            new_cap *= 2;
        }
        char* new_buff = realloc(sb->buff, new_cap);

        sb->buff = new_buff;
        sb->cap *= new_cap;

        memcpy(&sb->buff[sb->len], string, string_len);
        sb->len += string_len;
        sb->buff[sb->len] = '\0';

    }
}


void sb_free(StringBuilder* sb) {
    free(sb->buff);
    free(sb);
};