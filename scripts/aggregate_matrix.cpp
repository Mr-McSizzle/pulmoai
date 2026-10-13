#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

class FastIntegerReader {
 public:
  explicit FastIntegerReader(FILE* file) : file_(file), cursor_(0), end_(0) {}

  bool next(std::int64_t& value) {
    int character = nextCharacter();
    while (character != EOF && character <= ' ') character = nextCharacter();
    if (character == EOF) return false;

    bool negative = false;
    if (character == '-') {
      negative = true;
      character = nextCharacter();
    }
    if (character < '0' || character > '9') {
      throw std::runtime_error("unexpected non-integer in Matrix Market entries");
    }

    std::uint64_t parsed = 0;
    while (character >= '0' && character <= '9') {
      parsed = parsed * 10 + static_cast<unsigned>(character - '0');
      character = nextCharacter();
    }
    value = negative ? -static_cast<std::int64_t>(parsed)
                     : static_cast<std::int64_t>(parsed);
    return true;
  }

 private:
  int nextCharacter() {
    if (cursor_ == end_) {
      end_ = std::fread(buffer_, 1, sizeof(buffer_), file_);
      cursor_ = 0;
      if (end_ == 0) return EOF;
    }
    return static_cast<unsigned char>(buffer_[cursor_++]);
  }

  FILE* file_;
  char buffer_[1 << 20];
  std::size_t cursor_;
  std::size_t end_;
};

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: aggregate_matrix MATRIX.mtx CELL_DONOR_MAP OUTPUT.bin\n";
    return 2;
  }

  try {
    std::ifstream mapping(argv[2]);
    std::size_t cell_count = 0;
    std::size_t donor_count = 0;
    if (!(mapping >> cell_count >> donor_count) || cell_count == 0 || donor_count == 0) {
      throw std::runtime_error("invalid cell-to-donor map header");
    }
    std::vector<std::int32_t> donor_for_cell(cell_count);
    std::vector<std::uint64_t> cells_per_donor(donor_count, 0);
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
      if (!(mapping >> donor_for_cell[cell])) {
        throw std::runtime_error("cell-to-donor map ended early");
      }
      if (donor_for_cell[cell] < 0 ||
          static_cast<std::size_t>(donor_for_cell[cell]) >= donor_count) {
        throw std::runtime_error("cell-to-donor map contains an invalid donor index");
      }
      ++cells_per_donor[donor_for_cell[cell]];
    }
    std::int64_t extra_mapping_value = 0;
    if (mapping >> extra_mapping_value) {
      throw std::runtime_error("cell-to-donor map contains unexpected extra values");
    }

    FILE* matrix_file = std::fopen(argv[1], "rb");
    if (matrix_file == nullptr) throw std::runtime_error("could not open matrix file");
    char line[4096];
    if (std::fgets(line, sizeof(line), matrix_file) == nullptr ||
        std::strstr(line, "%%MatrixMarket matrix coordinate integer") == nullptr) {
      std::fclose(matrix_file);
      throw std::runtime_error("expected a coordinate integer Matrix Market file");
    }

    std::size_t feature_count = 0;
    std::size_t matrix_cell_count = 0;
    std::uint64_t expected_entries = 0;
    bool got_dimensions = false;
    while (std::fgets(line, sizeof(line), matrix_file) != nullptr) {
      if (line[0] == '%') continue;
      unsigned long long entries = 0;
      if (std::sscanf(line, "%zu %zu %llu", &feature_count, &matrix_cell_count,
                      &entries) != 3) {
        std::fclose(matrix_file);
        throw std::runtime_error("invalid Matrix Market dimensions");
      }
      expected_entries = static_cast<std::uint64_t>(entries);
      got_dimensions = true;
      break;
    }
    if (!got_dimensions || feature_count == 0 || matrix_cell_count != cell_count) {
      std::fclose(matrix_file);
      throw std::runtime_error("matrix dimensions do not match barcode metadata");
    }

    if (feature_count > std::numeric_limits<std::size_t>::max() / donor_count) {
      std::fclose(matrix_file);
      throw std::runtime_error("aggregated matrix is too large");
    }
    std::vector<std::uint64_t> totals(feature_count * donor_count, 0);
    FastIntegerReader reader(matrix_file);
    std::uint64_t mapped_entries = 0;
    for (std::uint64_t entry = 0; entry < expected_entries; ++entry) {
      std::int64_t feature = 0;
      std::int64_t cell = 0;
      std::int64_t count = 0;
      if (!reader.next(feature) || !reader.next(cell) || !reader.next(count)) {
        std::fclose(matrix_file);
        throw std::runtime_error("matrix ended before all declared entries were read");
      }
      if (feature < 1 || static_cast<std::size_t>(feature) > feature_count || cell < 1 ||
          static_cast<std::size_t>(cell) > cell_count || count < 0) {
        std::fclose(matrix_file);
        throw std::runtime_error("matrix entry is outside declared dimensions");
      }
      const std::size_t donor = static_cast<std::size_t>(donor_for_cell[cell - 1]);
      const std::size_t output_index =
          (static_cast<std::size_t>(feature) - 1) * donor_count + donor;
      totals[output_index] += static_cast<std::uint64_t>(count);
      ++mapped_entries;
      if ((entry + 1) % 100000000 == 0) {
        std::cerr << "Streamed " << (entry + 1) << " / " << expected_entries
                  << " matrix entries\n";
      }
    }
    std::int64_t extra_matrix_value = 0;
    if (reader.next(extra_matrix_value)) {
      std::fclose(matrix_file);
      throw std::runtime_error("matrix has more entries than its header declares");
    }
    if (std::ferror(matrix_file)) {
      std::fclose(matrix_file);
      throw std::runtime_error("error while reading the matrix file");
    }
    std::fclose(matrix_file);

    std::ofstream output(argv[3], std::ios::binary);
    if (!output) throw std::runtime_error("could not create aggregated output");
    output.write(reinterpret_cast<const char*>(totals.data()),
                 static_cast<std::streamsize>(totals.size() * sizeof(std::uint64_t)));
    if (!output) throw std::runtime_error("failed while writing aggregated output");

    std::cout << "features=" << feature_count << " donors=" << donor_count
              << " entries=" << expected_entries << " aggregated_entries=" << mapped_entries
              << "\n";
    for (std::size_t donor = 0; donor < donor_count; ++donor) {
      if (cells_per_donor[donor] == 0) {
        throw std::runtime_error("a donor has no cells in the barcode map");
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "aggregation failed: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
