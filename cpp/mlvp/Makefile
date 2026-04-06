CXX ?= c++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -pedantic -Iinclude

BUILD_DIR := build
BIN_DIR := $(BUILD_DIR)/bin
OBJ_DIR := $(BUILD_DIR)/obj

CORE_OBJ := $(OBJ_DIR)/core.o
APP_OBJ := $(OBJ_DIR)/main.o
TEST_OBJ := $(OBJ_DIR)/smoke_test.o
BENCHMARK_OBJ := $(OBJ_DIR)/benchmark.o
EVAL_MATRIX_OBJ := $(OBJ_DIR)/eval_matrix.o
FREEZE_CORPUS_OBJ := $(OBJ_DIR)/freeze_corpus.o
TUNE_WEIGHTS_OBJ := $(OBJ_DIR)/tune_weights.o

APP := $(BIN_DIR)/mlvp_demo
TEST := $(BIN_DIR)/smoke_test
BENCHMARK := $(BIN_DIR)/mlvp_benchmark
EVAL_MATRIX := $(BIN_DIR)/mlvp_eval_matrix
FREEZE_CORPUS := $(BIN_DIR)/mlvp_freeze_corpus
TUNE_WEIGHTS := $(BIN_DIR)/mlvp_tune_weights

.PHONY: all test clean

all: $(APP) $(BENCHMARK) $(EVAL_MATRIX) $(FREEZE_CORPUS) $(TUNE_WEIGHTS)

test: $(TEST)
	$(TEST)

clean:
	rm -rf $(BUILD_DIR)

$(BIN_DIR) $(OBJ_DIR):
	mkdir -p $@

$(CORE_OBJ): src/core.cpp include/mlvp/core.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(APP_OBJ): src/main.cpp include/mlvp/core.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TEST_OBJ): tests/smoke_test.cpp include/mlvp/core.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BENCHMARK_OBJ): src/benchmark.cpp include/mlvp/core.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(EVAL_MATRIX_OBJ): src/eval_matrix.cpp include/mlvp/core.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(FREEZE_CORPUS_OBJ): src/freeze_corpus.cpp include/mlvp/core.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TUNE_WEIGHTS_OBJ): src/tune_weights.cpp include/mlvp/core.hpp | $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(APP): $(CORE_OBJ) $(APP_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(TEST): $(CORE_OBJ) $(TEST_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BENCHMARK): $(CORE_OBJ) $(BENCHMARK_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(EVAL_MATRIX): $(CORE_OBJ) $(EVAL_MATRIX_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(FREEZE_CORPUS): $(CORE_OBJ) $(FREEZE_CORPUS_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

$(TUNE_WEIGHTS): $(CORE_OBJ) $(TUNE_WEIGHTS_OBJ) | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@
