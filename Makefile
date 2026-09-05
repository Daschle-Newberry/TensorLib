CC = gcc
CCFLAGS = \
	-std=c11 \
	-Iinclude	\
	-Wall	\
	-Wextra \
	-fanalyzer \
	-fsanitize=address \
	-g \
	-O0

LFLAGS = -shared

LIBNAME = tensorlib
TARGET = $(BUILDDIR)/lib$(LIBNAME).a

SRCDIR = src
TESTDIR = tests
BUILDDIR = build
TESTBUILDDIR = $(BUILDDIR)/tests

SRC = $(shell find $(SRCDIR) -name '*.c')
TESTSRC = $(shell find $(TESTDIR) -name '*.c')

TESTBIN = $(TESTSRC:$(TESTDIR)/%.c=$(TESTBUILDDIR)/%) 

OBJ = $(SRC:$(SRCDIR)/%.c=$(BUILDDIR)/%.o)
TESTOBJ = $(TESTSRC:$(TESTDIR)/%.c=$(BUILDDIR)/%.o)

$(TARGET): $(OBJ)
	ar rcs $@ $<

$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) -c $< -o $@

$(TESTBUILDDIR)/%: $(TESTDIR)/%.c $(TARGET)
	@mkdir -p $(dir $@)
	$(CC) $(CCFLAGS) $< -L$(BUILDDIR) -l$(LIBNAME) -o $@

test: $(TESTBIN)
	@for test in $(TESTBIN); do \
		echo "Running $$test"; \
		$$test || exit 1; \
	done

.PHONY:
clean:
	rm -rf $(BUILDDIR)

