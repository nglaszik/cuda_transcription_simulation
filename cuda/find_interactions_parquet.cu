#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <curand.h>
#include <curand_kernel.h>
#include <string>
#include <array>
#include <sstream>
#include <map>
#include <fstream>
#include <iterator>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <arrow/memory_pool.h>
#include <parquet/arrow/reader.h>
#include <memory>

using namespace std;

char path_input_matrix[200]="path_input_matrix";
char path_output_dir[200]="path_output_dir";
int batch_size = 0;
bool ranked = false;

typedef struct {
	double value;
	int original_index;
	double rank;
} DataPoint;

vector<vector<int>> cart_product (const vector<vector<int>>& v) {
	vector<vector<int>> s = {{}};
	for (const auto& u : v) {
		vector<vector<int>> r;
		for (const auto& x : s) {
			for (const auto y : u) {
				r.push_back(x);
				r.back().push_back(y);
			}
		}
		s = move(r);
	}
	return s;
}

int compare(const void *a, const void *b) {
	double diff = ((DataPoint *)a)->value - ((DataPoint *)b)->value;
	if (diff < 0) return -1;
	else if (diff > 0) return 1;
	return 0;
}

int getMaxGeneIndex(const int* array1, const int* array2, size_t size) {
	int max1 = *std::max_element(array1, array1 + size);
	int max2 = *std::max_element(array2, array2 + size);
	return std::max(max1, max2);
}

void convertCharrtoFarr(char *charr, float *farr) {
	int num_values = 1;
	int i;
	for(i=0; charr[i] != '\0'; i++){
		num_values += (charr[i] == ',');
	}
	// Traverse the charr
	int j = 0;
	char holder_arr[20]="";
	for (i = 0; charr[i] != '\0'; i++) {
		// if charr[i] is ',' then split
		if (charr[i] == ','){
			if (strcmp(holder_arr, "") != 0){
				farr[j] = atof(holder_arr);
				strcpy(holder_arr, "");
			}
			j++;
		}
		else {
			holder_arr[j] = charr[i];
		}
	}
}

long long int n_choose_r(int n, int r){
	
	int difference = n - r + 1;
	
	long long int r_factorial = 1;
	for(long long int i = (long long int)1; i <= r; i++){
		r_factorial = (long long int)r_factorial * (long long int)i;
	}
	
	long long int n_difference_factorial = 1;
	for(long long int i = (long long int)difference; i <= n; i++){
		n_difference_factorial = (long long int)n_difference_factorial * (long long int)i;
	}
	
	return (long long int)n_difference_factorial / (long long int)r_factorial;
	
}

int parseCommand(int argc, char **argv) {
	for(int i=1;i<argc;) {
		printf("argv[%u] = %s\n", i, argv[i]);
		if (strcmp(argv[i], "-i") == 0){
			strcpy(path_input_matrix, argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-o") == 0){
			strcpy(path_output_dir, argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-bs") == 0){
			batch_size=atoi(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "--ranked") == 0){
			ranked=true;
			i=i+1;
		}
		else{
			return 0;
		}
	}
	return 1;

}

// function to find correlations
__global__
void find_correlations(int *batch_pairs, int i_batch, int num_cells, int batch_size, int num_pairs_in_batch, double *batch_correlations, double *mrna_counts)
{
	//int n, float *x, float *y
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int i_pair = index; i_pair < num_pairs_in_batch; i_pair+=stride) {
		
		int i_gene_1 = batch_pairs[i_pair * 2];
		int i_gene_2 = batch_pairs[i_pair * 2 + 1];
		
		double sum_gene_1 = 0, sum_gene_2 = 0, sum_product = 0;
		double square_sum_gene_1 = 0, square_sum_gene_2 = 0;
		
		for (int i_cell = 0; i_cell < num_cells; i_cell++)
		{
			int i_gene_1_cell = i_gene_1 * num_cells + i_cell;
			int i_gene_2_cell = i_gene_2 * num_cells + i_cell;
			
			// sum of elements of array X.
			sum_gene_1 = sum_gene_1 + mrna_counts[i_gene_1_cell];
			// sum of elements of array Y.
			sum_gene_2 = sum_gene_2 + mrna_counts[i_gene_2_cell];
			// sum of X[i] * Y[i].
			sum_product = sum_product + mrna_counts[i_gene_1_cell] * mrna_counts[i_gene_2_cell];
			// sum of square of array elements.
			square_sum_gene_1 = square_sum_gene_1 + mrna_counts[i_gene_1_cell] * mrna_counts[i_gene_1_cell];
			square_sum_gene_2 = square_sum_gene_2 + mrna_counts[i_gene_2_cell] * mrna_counts[i_gene_2_cell];
		}
	  
		// use formula for calculating correlation coefficient.
		double corr = (double)((double)num_cells * (double)sum_product - (double)sum_gene_1 * (double)sum_gene_2)  / sqrt((double)((double)num_cells * (double)square_sum_gene_1 - (double)sum_gene_1 * (double)sum_gene_1) * (double)((double)num_cells * (double)square_sum_gene_2 - (double)sum_gene_2 * (double)sum_gene_2));
		
		batch_correlations[i_pair] = corr;
		
	}
}

// /usr/local/cuda-12.6/bin/nvcc -std=c++17 /home/data/nlaszik/cuda_simulation/code/cuda/find_interactions_parquet.cu -o /home/data/nlaszik/cuda_simulation/code/cuda/build/find_interactions_parquet -lparquet -larrow -lineinfo -lz


// WT
// /home/data/nlaszik/cuda_simulation/code/cuda/build/find_interactions_parquet -bs 50000000 -i /home/data/Shared/shared_datasets/sc_rna_seq/nlaszik/wt_tko_downsampled_mito/transcript_counts/wt_rep2.filtered.normalized.gmauto.parquet.gz -o /home/data/Shared/shared_datasets/sc_rna_seq/nlaszik/wt_tko_downsampled_mito/gene_correlations_wt/ --ranked

// DKO
// /home/data/nlaszik/cuda_simulation/code/cuda/build/find_interactions_parquet -bs 50000000 -i /home/data/Shared/shared_datasets/sc_rna_seq/nlaszik/wt_dko_downsampled_mito/transcript_counts/dko_rep2.filtered.normalized.gmauto.parquet.gz -o /home/data/Shared/shared_datasets/sc_rna_seq/nlaszik/wt_dko_downsampled_mito/gene_correlations_dko/ --ranked

// TKO
// /home/data/nlaszik/cuda_simulation/code/cuda/build/find_interactions_parquet -bs 50000000 -i /home/data/Shared/shared_datasets/sc_rna_seq/nlaszik/wt_tko_downsampled_mito/transcript_counts/tko_rep2.filtered.normalized.gmauto.parquet.gz -o /home/data/Shared/shared_datasets/sc_rna_seq/nlaszik/wt_tko_downsampled_mito/gene_correlations/

int main(int argc, char** argv)
{
	
	if(!parseCommand(argc, argv)) {
		cout<<"Error in arguments..\n";
		exit(0);
	}
	
	int *batch_pairs;
	double *batch_correlations, *mrna_counts;
	
	// get memory free on gpu
	float free_m,total_m,used_m;
	size_t free_t,total_t;
	cudaMemGetInfo(&free_t,&total_t);
	free_m =(float)free_t/1048576.0;
	total_m=(float)total_t/1048576.0;
	used_m=total_m-free_m;
	printf ("mem free %f MB, mem total %f MB, mem used %f MB\n", free_m, total_m, used_m);
	
	// Reading a Parquet file
	std::shared_ptr<arrow::io::ReadableFile> infile;
	PARQUET_ASSIGN_OR_THROW(
		infile,
		arrow::io::ReadableFile::Open(path_input_matrix, arrow::default_memory_pool())
	);
	
	// Create a Parquet reader
	std::unique_ptr<parquet::arrow::FileReader> reader;
	PARQUET_THROW_NOT_OK(
		parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader)
	);
	
	// Read the table
	std::shared_ptr<arrow::Table> table;
	PARQUET_THROW_NOT_OK(reader->ReadTable(&table));
	
	// Extract data
	auto num_cells = table->num_rows();
	auto num_genes = table->num_columns() - 1;
	
	cudaMallocManaged(&mrna_counts, num_genes * num_cells * sizeof(double));
	vector<string> gene_names;
	
	// Assume the data is of type float
	int idx = 0;
	for (int i_gene = 0; i_gene < num_genes; ++i_gene) {
				
		// count data... cast to float
		std::shared_ptr<arrow::Array> column_array = table->column(i_gene)->chunk(0);
		
		// Check data type and cast appropriately
		if (column_array->type_id() == arrow::Type::DOUBLE) {
			gene_names.push_back(table->field(i_gene)->name());
			auto double_array = std::static_pointer_cast<arrow::DoubleArray>(column_array);
			
			if (ranked){
				// if spearman, assign rank instead of count
				DataPoint data_points[num_cells];
				
				// Store original values and indices
				for (int i_cell = 0; i_cell < num_cells; i_cell++) {
					data_points[i_cell].value = double_array->Value(i_cell);
					data_points[i_cell].original_index = i_cell;
				}
				
				// Sort by value
				qsort(data_points, num_cells, sizeof(DataPoint), compare);
				
				// Assign ranks, handling ties by assigning the average rank
				for (int i = 0; i < num_cells; i++) {
					int start = i;
					while (i < num_cells - 1 && data_points[i].value == data_points[i + 1].value) {
						i++;
					}
					double rank = (start + i + 2) / 2; // Average rank for ties
					for (int j = start; j <= i; j++) {
						int i_cell = data_points[j].original_index;
						// Store rank in 1D array
						mrna_counts[i_gene * num_cells + i_cell] = rank;
						//printf("gene %i cell %i rank=%f\n", i_gene, i_cell, rank);
					}
				}
			}
			else {
				for (int i = 0; i < num_cells; i++) {
					mrna_counts[idx++] = double_array->Value(i);
				}
			}
		}
	}
	
	printf("number of genes: %i\n", num_genes);
	printf("number of genes actually found: %lu\n", gene_names.size());
	printf("number of cells: %ld\n", num_cells);
	
	cudaMallocManaged(&batch_pairs, batch_size * 2 * sizeof(int));
	cudaMallocManaged(&batch_correlations, batch_size * sizeof(double));
	
	printf("creating combinations vector...\n");
	long long int num_gene_pairs = n_choose_r(num_genes, 2);
	int i_gene_pair = 0;
	
	// initialize large array
	int ** gene_pairs;
	gene_pairs = (int**)malloc(sizeof(int*)*num_gene_pairs);
	for (int i_gene = 0; i_gene < num_gene_pairs; i_gene++){
		gene_pairs[i_gene] = (int*)malloc(sizeof(int)*2);
	}
	
	for (int i_gene_1 = 0; i_gene_1 < num_genes; i_gene_1++){
		for (int i_gene_2 = 0; i_gene_2 < num_genes; i_gene_2++){
			// easy way to get combinations... upper triangle can be selected by index comparison
			if (i_gene_2 > i_gene_1){
				gene_pairs[i_gene_pair][0] = i_gene_1;
				gene_pairs[i_gene_pair][1] = i_gene_2;
				i_gene_pair++;
			}
		}
	}
	
	int num_batches;
	if (num_gene_pairs <= batch_size) {
		batch_size = num_gene_pairs;
		num_batches = 1;
	}
	else {
		num_batches = (int)ceil(num_gene_pairs / batch_size) + 1;
	}
	
	printf("number of gene pairs: %llu, num batches: %i, final batch size: %i\n", num_gene_pairs, num_batches, batch_size);
	
	double *correlations = new double[num_gene_pairs];

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	// setup and allocate memory for curand
	int N = batch_size;
	int blockSize = 32;
	int numBlocks = (N + blockSize - 1) / blockSize;
	
	cudaEventRecord(start);
	
	for (int i_batch = 0; i_batch < num_batches; i_batch++){
		
		// assign pairs vector to gpu memory in batches
		int i_batch_pair = 0;
		for (long long int i_pair = (long long int)i_batch * (long long int)batch_size; i_pair < (long long int)(i_batch + 1) * (long long int)batch_size; i_pair++){
			if (i_pair < num_gene_pairs) {
				batch_pairs[i_batch_pair * 2] = gene_pairs[i_pair][0];
				batch_pairs[i_batch_pair * 2 + 1] = gene_pairs[i_pair][1];
				i_batch_pair++;
			}
			else {
				break;
			}
		}
		
		printf("processing batch %i, num pairs: %i...\n", i_batch + 1, i_batch_pair);
		
		find_correlations<<<numBlocks, blockSize>>>(batch_pairs, i_batch, num_cells, batch_size, i_batch_pair, batch_correlations, mrna_counts);
		
		cudaEventRecord(stop);
		cudaDeviceSynchronize();
		
		cudaError_t err = cudaGetLastError();
		if (err != cudaSuccess) {
			printf("CUDA Error: %s\n", cudaGetErrorString(err));
		}
		
		// transfer batch correlations to all correlations
		for (int i_batch_pair_processed = 0; i_batch_pair_processed < i_batch_pair; i_batch_pair_processed++){
			long long int i_pair = (long long int)i_batch * (long long int)batch_size + (long long int)i_batch_pair_processed;
			correlations[i_pair] = batch_correlations[i_batch_pair_processed];
		}
		
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		
		printf("Elapsed seconds: %f\n", milliseconds/1000);
		
	}
	
	//output file to csv
	
	FILE *outfile;
	string path_output_string;
	if (ranked) {
		path_output_string = string(path_output_dir) + "ranked_correlations.csv";
	}
	else {
		path_output_string = string(path_output_dir) + "correlations.csv";
	}
	const char *path_output = path_output_string.c_str();
	
	printf("%s\n", path_output);
	
	outfile = fopen(path_output, "w");//create a file
	fprintf(outfile, "gene_1,gene_2,correlation\n");
	for (long long int i_pair = 0; i_pair < num_gene_pairs; i_pair++){
		fprintf(outfile, "%s,%s,%.16f\n", gene_names[gene_pairs[i_pair][0]].c_str(), gene_names[gene_pairs[i_pair][1]].c_str(), correlations[i_pair]);
	}
	fclose(outfile);
	
	// Free memory
	delete [] correlations;
	cudaFree(mrna_counts);
	cudaFree(batch_pairs);
	cudaFree(batch_correlations);
	return 0;
}




