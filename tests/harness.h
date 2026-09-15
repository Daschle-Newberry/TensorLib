#ifndef HARNESS_H
#define HARNESS_H

#include <stdio.h>

#define RUN(name) do{ \
  printf("\e[1;1;24;35mRunning test for:\e[0m %s\n", #name);\
  test_##name(); \
} while(0)
#define TEST(name) \
    static void test_##name(void)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, \
            "   \e[1;31mFAIL\e[0m: %s\n    at %s:%d\n", \
             #cond, __FILE__, __LINE__); \
    } \
    else {\
        fprintf(stdout, \
            "   \e[1;32mPASS\e[0m: %s\n", #cond); \
    }\
} while (0)

#define ASSERT_ARRAY_EQ(actual, expected, len, T) do { \
  T* _exp = (expected); \
  size_t _len = (len);  \
  int _ok = 1; \
  for(size_t _i = 0; _i < _len; _i++) { \
    if((actual)[_i] != _exp[_i]) { \
        _ok = 0; \
        break; \
    } \
  } \
  if (!_ok) { \
      fprintf(stderr, \
          "   \e[1;31mFAIL\e[0m: %s == %s\n    at %s:%d\n", \
           #actual, #expected, __FILE__, __LINE__); \
  } \
  else { \
      fprintf(stdout, \
          "   \e[1;32mPASS\e[0m: %s == %s\n", \
           #actual, #expected); \
  } \
}while(0)

#endif
