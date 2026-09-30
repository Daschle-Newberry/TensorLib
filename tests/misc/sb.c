#include "harness.h"
#include "private/string_builder.h"



TEST(sb_append) {
  StringBuilder sb;
  sb_init(&sb);

  sb_append(&sb, "Hello");
  ASSERT(!strcmp("Hello", sb.buf));

  sb_append(&sb, ", World!");
  ASSERT(!strcmp("Hello, World!", sb.buf));
  ASSERT(sb.len = 13);

  free(sb.buf);
  sb_init(&sb);
  
  char s3[sb.cap];
  memset(s3, '.', sizeof(s3));

  s3[sb.cap - 1] = '\0';
  sb_append(&sb, s3);

  ASSERT(sb.cap == 128);

  sb_append(&sb, "1234");
  ASSERT(sb.cap > 128);

  free(sb.buf);
}


int main() {
  RUN(sb_append);
  return 0;
}





