#ifndef HARNESS_H
#define HARNESS_H

#include <stdio.h>

#define RUN(name) test_##name()

#define TEST(name) \
    static void test_##name(void)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, \
            "\e[1;31mFAIL\e[0m: %s\n    at %s:%d\n", \
             #cond, __FILE__, __LINE__); \
    } \
    else {\
        fprintf(stderr, \
            "\e[1;32mPASS\e[0m: %s\n", #cond); \
    }\
} while (0)

#endif
