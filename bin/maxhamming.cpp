#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_set>
#include <limits>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <iomanip>
#include <chrono>
#include <random>

using namespace std;
using namespace std::chrono;

// -------------------- Data Structures --------------------

struct FastaSeq {
    string header;
    string sequence;
};


struct SelectionResult {
    vector<FastaSeq> sequences;
    vector<int> indices; 
};

// -------------------- Utility Functions --------------------

double SequenceDistance(const string& a, const string& b, bool ignore_gap = true) {
    if (a.size() != b.size())
        throw runtime_error("Sequences are not aligned or of different length.");

    int diff = 0, count = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (ignore_gap && (a[i] == '-' || b[i] == '-')) continue;
        if (a[i] != b[i]) ++diff;
        ++count;
    }
    return count > 0 ? static_cast<double>(diff) / count : 0.0;
}

vector<FastaSeq> ReadFASTA(const string& filename, bool verbose) {
    auto start = high_resolution_clock::now();
    ifstream infile(filename);
    if (!infile) throw runtime_error("Cannot open input file: " + filename);

    vector<FastaSeq> sequences;
    string line, header, sequence;
    size_t seq_count = 0;

    while (getline(infile, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            if (!sequence.empty()) {
                sequences.push_back({header, sequence});
                sequence.clear();
                seq_count++;
            }
            header = line;
        } else {
            sequence += line;
        }
    }
    if (!sequence.empty()) {
        sequences.push_back({header, sequence});
        seq_count++;
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    if (verbose) {
        cerr << "Loaded " << seq_count << " sequences in " 
             << duration.count() << " ms" << endl;
    }
    return sequences;
}

void WriteFASTA(const vector<FastaSeq>& sequences, const string& filename, bool verbose) {
    auto start = high_resolution_clock::now();
    ofstream out(filename);
    if (!out) throw runtime_error("Cannot write output file: " + filename);
    
    size_t total_chars = 0;
    for (const auto& s : sequences) {
        out << s.header << '\n';
        size_t w = 60;
        for (size_t i = 0; i < s.sequence.size(); i += w) {
            out << s.sequence.substr(i, min(w, s.sequence.size() - i)) << '\n';
        }
        total_chars += s.sequence.size();
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    if (verbose) {
        cerr << "Wrote " << sequences.size() << " sequences (" 
             << total_chars << " characters) in " 
             << duration.count() << " ms" << endl;
    }
}


void WriteIndices(const vector<int>& indices, const string& filename, bool verbose) {
    auto start = high_resolution_clock::now();
    ofstream out(filename);
    if (!out) throw runtime_error("Cannot write output file: " + filename);
    
    for (int idx : indices) {
        out << idx << '\n';
    }
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);
    
    if (verbose) {
        cerr << "Wrote " << indices.size() << " indices in " 
             << duration.count() << " ms" << endl;
    }
}

// -------------------- Core Selection Function --------------------


SelectionResult SelectDiverseSequences(
    const vector<FastaSeq>& all_seqs, int N,
    bool ignore_gap, double min_dist_threshold, 
    bool verbose, bool random_seed = false
) {
    auto total_start = high_resolution_clock::now();
    int M = all_seqs.size();
    
   
    SelectionResult result;
    vector<FastaSeq>& selected = result.sequences;
    vector<int>& selected_indices_in_order = result.indices;
    
    unordered_set<int> selected_indices;
    vector<double> min_distance(M, 1.0);
    int total_selected = 0;
    bool threshold_reached = false;

    if (M <= N) {
        if (verbose) {
            cerr << "Input sequence count (" << M << ") <= N (" << N << "), returning all sequences" << endl;
        }
     
        for (int i = 0; i < M; i++) {
            selected.push_back(all_seqs[i]);
            selected_indices_in_order.push_back(i);
        }
        return result;
    }

 
    int seed_idx = 0;
    if (random_seed) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<> dis(0, M - 1);
        seed_idx = dis(gen);
    }

    auto select_start = high_resolution_clock::now();
    selected.push_back(all_seqs[seed_idx]);
    selected_indices_in_order.push_back(seed_idx);
    selected_indices.insert(seed_idx);
    total_selected++;
    
    if (verbose) {
        const string& header = all_seqs[seed_idx].header;
        const char* header_preview = header.c_str() + 1;
        cerr << "[1/" << N << "] Selected: ";
        if (header.length() > 51) {
            cerr.write(header_preview, 50);
            cerr << "...";
        } else {
            cerr << header_preview;
        }
        cerr << " (Index: " << seed_idx << ")" << endl;
        auto select_end = high_resolution_clock::now();
        auto select_duration = duration_cast<milliseconds>(select_end - select_start);
        cerr << "Initial selection completed in " << select_duration.count() << " ms" << endl;
    }

 
    if (verbose) {
        cerr << "Initializing min distances for " << M-1 << " sequences..." << endl;
    }
    auto init_start = verbose ? high_resolution_clock::now() : time_point<high_resolution_clock>{};
    for (int i = 0; i < M; ++i) {
        if (i == seed_idx) continue;
        min_distance[i] = SequenceDistance(
            all_seqs[i].sequence, 
            all_seqs[seed_idx].sequence, 
            ignore_gap
        );
    }
    auto init_end = verbose ? high_resolution_clock::now() : time_point<high_resolution_clock>{};
    if (verbose) {
        auto init_duration = duration_cast<milliseconds>(init_end - init_start);
        cerr << "Distance initialization completed in " << init_duration.count() << " ms" << endl;
    }
    
 
    if (verbose) {
        cerr << "Beginning greedy selection of " << N-1 << " sequences" 
             << (min_dist_threshold > 0 ? " with min_dist threshold: " + to_string(min_dist_threshold) : "")
             << "..." << endl;
    }
    
    for (int count = 1; count < N; count++) {
        auto iter_start = verbose ? high_resolution_clock::now() : time_point<high_resolution_clock>{};
        
  
        auto scan_start = verbose ? high_resolution_clock::now() : time_point<high_resolution_clock>{};
        int best_idx = -1;
        double best_min_dist = -1.0;
        
        for (int i = 0; i < M; i++) {
            if (selected_indices.count(i)) continue;
            
            if (best_idx == -1 || min_distance[i] > best_min_dist) {
                best_min_dist = min_distance[i];
                best_idx = i;
            }
        }
        auto scan_end = verbose ? high_resolution_clock::now() : time_point<high_resolution_clock>{};
        auto scan_duration = verbose ? duration_cast<milliseconds>(scan_end - scan_start) : milliseconds{};
        
    
        if (min_dist_threshold > 0 && best_min_dist <= min_dist_threshold) {
            if (verbose) {
                cerr << "Stopping early at iteration " << count 
                     << ": best_min_dist " << fixed << setprecision(4) << best_min_dist
                     << " <= threshold " << min_dist_threshold << endl;
            }
            threshold_reached = true;
            break;
        }
        
  
        if (best_idx == -1) {
            if (verbose) {
                cerr << "Warning: No valid candidate found in scan, using fallback" << endl;
            }
            for (int i = 0; i < M; i++) {
                if (!selected_indices.count(i)) {
                    best_idx = i;
                    best_min_dist = min_distance[i];
                    break;
                }
            }
        }
        

        auto add_start = verbose ? high_resolution_clock::now() : time_point<high_resolution_clock>{};
        selected.push_back(all_seqs[best_idx]);
        selected_indices_in_order.push_back(best_idx); 
        selected_indices.insert(best_idx);
        total_selected++;
        auto add_end = verbose ? high_resolution_clock::now() : time_point<high_resolution_clock>{};
        auto add_duration = verbose ? duration_cast<milliseconds>(add_end - add_start) : milliseconds{};
        

        if (verbose) {
            const string& header = all_seqs[best_idx].header;
            const char* header_preview = header.c_str() + 1; 
            cerr << "[" << total_selected << "/" << N << "] Selected: ";
            if (header.length() > 51) {
                cerr.write(header_preview, 50);
                cerr << "...";
            } else {
                cerr << header_preview;
            }
            cerr << " | Index: " << best_idx
                 << " | MinDist: " << fixed << setprecision(4) << best_min_dist
                 << " | Scan time: " << scan_duration.count() << " ms";
        }
        

        auto update_start = verbose ? high_resolution_clock::now() : time_point<high_resolution_clock>{};
        size_t updated = 0;
        for (int i = 0; i < M; ++i) {
            if (selected_indices.count(i)) continue;
            
            double new_dist = SequenceDistance(
                all_seqs[i].sequence, 
                all_seqs[best_idx].sequence, 
                ignore_gap
            );
            
            if (new_dist < min_distance[i]) {
                min_distance[i] = new_dist;
                if (verbose) updated++;
            }
        }
        auto update_end = verbose ? high_resolution_clock::now() : time_point<high_resolution_clock>{};
        if (verbose) {
            auto update_duration = duration_cast<milliseconds>(update_end - update_start);
            auto iter_duration = duration_cast<milliseconds>(update_end - iter_start);
            
            cerr << " | Updated: " << updated 
                 << " | Update time: " << update_duration.count() << " ms"
                 << " | Add time: " << add_duration.count() << " ms"
                 << " | Iter total: " << iter_duration.count() << " ms" << endl;
        }
    }
    
    auto total_end = verbose ? high_resolution_clock::now() : time_point<high_resolution_clock>{};
    if (verbose) {
        auto total_duration = duration_cast<milliseconds>(total_end - total_start);
        cerr << "Selection completed in " << total_duration.count() << " ms" << endl;
        cerr << "Total sequences processed: " << M << endl;
        cerr << "Selected sequences: " << selected.size() << " of " << N;
        if (threshold_reached) {
            cerr << " (early stop due to min_dist threshold)";
        }
        cerr << endl;
        cerr << "Selected indices: ";
        for (size_t i = 0; i < selected_indices_in_order.size(); ++i) {
            if (i > 0) cerr << ", ";
            cerr << selected_indices_in_order[i];
        }
        cerr << endl;
        cerr << "Average min distance: " << fixed << setprecision(4);
        if (selected.size() > 1) {
            double sum = 0.0;
            for (size_t i = 1; i < selected.size(); ++i) {
                sum += min_distance[selected_indices_in_order[i]];
            }
            cerr << (sum / (selected.size() - 1));
        } else {
            cerr << "N/A";
        }
        cerr << endl;
    }

    return result;
}

// -------------------- Command-line Interface --------------------

void PrintUsage(const char* prog) {
    cerr << "Usage: " << prog << " -i input.fa -o output.fa -n N [-ig] [-t threshold] [-f format] [-r] [-v]\n"
         << "Options:\n"
         << "  -i     Input FASTA file (required)\n"
         << "  -o     Output file (required)\n"
         << "  -n     Number of sequences to retain (required, >0)\n"
         << "  -ig    Ignore gap characters when computing distance (optional)\n"
         << "  -t     MinDist threshold for early stopping (optional, 0.0-1.0)\n"
         << "  -f     Output format: 'msa' (FASTA) or 'index' (indices) [default: msa]\n"
         << "  -r     Use random seed selection instead of first sequence (optional)\n"
         << "  -v     Enable verbose logging (optional)\n"
         << "  -h, --help    Show this help message\n";
}

// -------------------- Main --------------------

int main(int argc, char* argv[]) {
    auto main_start = high_resolution_clock::now();
    
    string input_file, output_file;
    int N = 0;
    bool ignore_gap = false;
    bool verbose = false;
    bool random_seed = false;
    double min_dist_threshold = 0.0;
    string output_format = "msa";


    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if ((arg == "-h") || (arg == "--help")) {
            PrintUsage(argv[0]);
            return 0;
        } else if (arg == "-i" && i + 1 < argc) {
            input_file = argv[++i];
        } else if (arg == "-o" && i + 1 < argc) {
            output_file = argv[++i];
        } else if (arg == "-n" && i + 1 < argc) {
            N = stoi(argv[++i]);
            if (N <= 0) {
                cerr << "Error: N must be positive integer" << endl;
                return 1;
            }
        } else if (arg == "-ig") {
            ignore_gap = true;
        } else if (arg == "-t" && i + 1 < argc) {
            min_dist_threshold = stod(argv[++i]);
            if (min_dist_threshold < 0 || min_dist_threshold > 1.0) {
                cerr << "Error: Threshold must be between 0.0 and 1.0" << endl;
                return 1;
            }
        } else if (arg == "-f" && i + 1 < argc) {
            output_format = argv[++i];
            if (output_format != "msa" && output_format != "index") {
                cerr << "Error: Output format must be 'msa' or 'index'" << endl;
                return 1;
            }
        } else if (arg == "-r") {
            random_seed = true;
        } else if (arg == "-v") {
            verbose = true;
        } else {
            cerr << "Unknown or incomplete argument: " << arg << endl;
            PrintUsage(argv[0]);
            return 1;
        }
    }


    if (input_file.empty() || output_file.empty() || N <= 0) {
        PrintUsage(argv[0]);
        return 1;
    }

    try {
        if (verbose) {
            cerr << "Reading sequences from: " << input_file << endl;
        }
        auto sequences = ReadFASTA(input_file, verbose);
        if (verbose) {
            cerr << "Loaded " << sequences.size() << " sequences" << endl;
            cerr << "Selecting " << N << " diverse sequences";
            if (min_dist_threshold > 0) {
                cerr << " with min_dist threshold: " << min_dist_threshold;
            }
            if (random_seed) {
                cerr << " with random seed selection";
            }
            cerr << endl;
        }
        
        auto result = SelectDiverseSequences(sequences, N, ignore_gap, min_dist_threshold, verbose, random_seed);
        
        if (verbose) {
            cerr << "Output format: " << output_format << endl;
        }
        

        if (output_format == "msa") {
            if (verbose) {
                cerr << "Writing " << result.sequences.size() 
                     << " sequences to: " << output_file << endl;
            }
            WriteFASTA(result.sequences, output_file, verbose);
        } else if (output_format == "index") {
            if (verbose) {
                cerr << "Writing " << result.indices.size() 
                     << " indices to: " << output_file << endl;
            }
            WriteIndices(result.indices, output_file, verbose);
        }
        
        auto main_end = high_resolution_clock::now();
        auto main_duration = duration_cast<milliseconds>(main_end - main_start);
        

        cout << "Input sequences: " << sequences.size() << endl;
        cout << "Output sequences: " << result.sequences.size() << endl;
        if (output_format == "index") {
            cout << "Output indices: ";
            for (int i = 0; i < result.indices.size(); i++) {
                if (i > 0) cout << ", ";
                cout << result.indices[i];
            }
            cout << endl;
        }
        cout << "Total time: " << main_duration.count() << " ms" << endl;
        
        if (result.sequences.size() < N && min_dist_threshold > 0) {
            cout << "Note: Stopped early at " << result.sequences.size() 
                 << " sequences due to min_dist threshold" << endl;
        }
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}
