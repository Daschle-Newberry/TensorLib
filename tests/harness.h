#define RUN(name) test_##name()

#define TEST(name) \
    static void test_##name(void)

#define ASSERT(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, \
            "FAIL: %s:%d: %s\n", \
            __FILE__, __LINE__, #cond); \
        return false; \
    } \
} while (0)
