CC = gcc
CCFLAGS = -std=c11 -g -fsanitize=address -Iinclude

LIB_NAME = TensorLib
LIB_FILE = build/lib${LIB_NAME}.a

EXEC = build/TensorLibTest

SRC_DIR = src
BUILD_DIR = build
TEST_DIR = test

LIB_SRC = ${wildcard ${SRC_DIR}/*.c}
LIB_OBJS = ${patsubst ${SRC_DIR}/%.c,${BUILD_DIR}/%.o, ${LIB_SRC}} 

TEST_SRC = ${wildcard ${TEST_DIR}/*.c}
TEST_OBJS = ${patsubst ${TEST_DIR}/%.c,${BUILD_DIR}/%.o, ${TEST_SRC}} 

${LIB_FILE}: ${LIB_OBJS}
	ar rcs $@ ${LIB_OBJS}

${EXEC}: ${LIB_FILE} ${TEST_OBJS}
	${CC} ${CCFLAGS} ${TEST_OBJS} ${LIB_FILE} -o $@

${BUILD_DIR}/%.o: ${SRC_DIR}/%.c
	${CC} ${CCFLAGS} -c $< -o $@

${BUILD_DIR}/%.o: ${TEST_DIR}/%.c
	${CC} ${CCFLAGS} -c $< -o $@

run: ${EXEC}
	./${EXEC}

clean:
	rm -f ${BUILD_DIR}/*.o ${LIB_FILE} ${EXEC}
