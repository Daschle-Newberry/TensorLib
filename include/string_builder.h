#ifndef STRING_BUILDER_H
#define STRING_BUILDER_H

typedef struct {
    char* buff;
    size_t len;
    size_t cap;
} StringBuilder;

void init_sb(StringBuilder* sb);
void sb_append(StringBuilder* sb, const char* string);
void sb_free(StringBuilder* sb);

#endif //STRING_BUILDER_H