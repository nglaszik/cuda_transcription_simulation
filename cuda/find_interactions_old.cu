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

using namespace std;

char path_input_matrix[200]="path_input_matrix";
char path_output_dir[200]="path_output_dir";
int batch_size = 0;

void pearson(double *X, double *Y, int *pnrx,  int *pnry, int *pnc, double *r, double *t)
{
    double *sX;     // sum of X
    double *sY;     // sum of Y
    double *sX2;    // sum of X^2
    double *sY2;    // sum of Y^2
    double **sXY;   // sum of X*Y

    double sqrtN2;

    int i,j,k;
    int Nrx,Nry,Nc;
    
    Nrx=pnrx[0]; // number of rows (vectors) in matrix X
    Nry=pnry[0]; // number of rows (vectors) in matrix Y
    Nc=pnc[0];   // number of pairs per vector

    // memory allocation
    sX = (double *) calloc(1,Nrx*sizeof(double));
    sY = (double *) calloc(1,Nry*sizeof(double));
    sX2 = (double *) calloc(1,Nrx*sizeof(double));
    sY2 = (double *) calloc(1,Nry*sizeof(double));
    sXY = (double **) malloc(Nrx*sizeof(double*));
    for (i=0;i<Nrx;i++)
        sXY[i] = (double *) calloc(1,Nry*sizeof(double));
    
    // sum of X and X^2
    for (i=0;i<Nrx;i++) {
        for (k=0;k<Nc;k++) {
            sX[i] = sX[i] + X[i*Nc+k];
            sX2[i] = sX2[i] + (X[i*Nc+k] * X[i*Nc+k]);
        }
    }

    // sum of Y and Y^2
    for (j=0;j<Nry;j++) {
        for (k=0;k<Nc;k++) {
            sY[j] = sY[j] + Y[j*Nc+k];
            sY2[j] = sY2[j] + (Y[j*Nc+k] * Y[j*Nc+k]);
        }
    }

    // sum of X*Y
    for (i=0;i<Nrx;i++) {
        for (j=0;j<Nry;j++) {
            for (k=0;k<Nc;k++) {
                sXY[i][j] = sXY[i][j] + (X[i*Nc+k] * Y[j*Nc+k]);
            }
        }
    }
    
    sqrtN2 = sqrt((double)(Nc-2));
    for (i=0;i<Nrx;i++) {
        for (j=0;j<Nry;j++) {
            // Pearson's r
            r[i*Nry+j] = (Nc*sXY[i][j] - (sX[i]*sY[j])) / ( sqrt(Nc*sX2[i] - (sX[i]*sX[i])) * sqrt(Nc*sY2[j] - (sY[j]*sY[j])) );
            // t-value
            t[i*Nry+j] = sqrtN2 * r[i*Nry+j] / sqrt( (double)1.f - (r[i*Nry+j]*r[i*Nry+j]) );
        }
    }
    
    // free memory allocated
    free(sX);
    free(sY);
    free(sX2);
    free(sY2);
    for (i=0;i<Nrx;i++)
        free(sXY[i]);
    free(sXY);
    
}

float pearson_correlation(int X[], int Y[], int n)
{
  
    int sum_X = 0, sum_Y = 0, sum_XY = 0;
    int squareSum_X = 0, squareSum_Y = 0;
  
    for (int i = 0; i < n; i++)
    {
        // sum of elements of array X.
        sum_X = sum_X + X[i];
  
        // sum of elements of array Y.
        sum_Y = sum_Y + Y[i];
  
        // sum of X[i] * Y[i].
        sum_XY = sum_XY + X[i] * Y[i];
  
        // sum of square of array elements.
        squareSum_X = squareSum_X + X[i] * X[i];
        squareSum_Y = squareSum_Y + Y[i] * Y[i];
    }
  
    // use formula for calculating correlation coefficient.
    double corr = (double)(n * sum_XY - sum_X * sum_Y)
                  / sqrt((n * squareSum_X - sum_X * sum_X)
                      * (n * squareSum_Y - sum_Y * sum_Y));
  
    return corr;
}

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
		else{
			return 0;
		}
    }
    return 1;

}

// function to find correlations
__global__
void find_correlations(int *batch_pairs, int i_batch, int num_cells, int batch_size, int num_pairs_in_batch, double *batch_correlations, int *mrna_counts)
{
	//int n, float *x, float *y
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int i_pair = index; i_pair < num_pairs_in_batch; i_pair+=stride) {
		
		int i_gene_1 = batch_pairs[i_pair * 2];
		int i_gene_2 = batch_pairs[i_pair * 2 + 1];
		
		int sum_gene_1 = 0, sum_gene_2 = 0, sum_product = 0;
	    int square_sum_gene_1 = 0, square_sum_gene_2 = 0;
	  
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

// nvcc -std=c++17 /home/data/nlaszik/cuda_simulation/code/cuda/find_interactions.cu -o /home/data/nlaszik/cuda_simulation/code/cuda/build/find_interactions -lineinfo -lz


// SRP215251
// /home/data/nlaszik/cuda_simulation/code/cuda/build/find_interactions -bs 50000000 -i /home/data/Shared/shared_datasets/sc-rna-seq/SRP215251/figures/wt/tpm_filtered.csv -o /home/data/nlaszik/cuda_simulation/output/SRP215251/WT/

// /home/data/nlaszik/cuda_simulation/code/cuda/build/find_interactions -bs 50000000 -i /home/data/Shared/shared_datasets/sc-rna-seq/SRP215251/figures/dko/tpm_filtered.csv -o /home/data/nlaszik/cuda_simulation/output/SRP215251/DKO/

// SRP313343
// /home/data/nlaszik/cuda_simulation/code/cuda/build/find_interactions -bs 50000000 -i /home/data/Shared/shared_datasets/sc-rna-seq/SRP313343/seurat/transcript_counts/srr14139729_transcript_counts.t.filtered.csv -o /home/data/nlaszik/cuda_simulation/output/SRP313343/SRR14139729/

// /home/data/nlaszik/cuda_simulation/code/cuda/build/find_interactions -bs 50000000 -i /home/data/Shared/shared_datasets/sc-rna-seq/SRP313343/seurat/transcript_counts/srr14139730_transcript_counts.t.filtered.csv -o /home/data/nlaszik/cuda_simulation/output/SRP313343/SRR14139730/

// SRP299892 normalized
// smaller batch size due to higher # of cells?
// /home/data/nlaszik/cuda_simulation/code/cuda/build/find_interactions -bs 50000000 -i /home/data/Shared/shared_datasets/sc_rna_seq/SRP299892/seurat/transcript_counts/srr13336770_transcript_counts.filtered.norm.csv -o /home/data/nlaszik/cuda_simulation/figures/SRP299892/interactions/

int main(int argc, char** argv)
{
	
	if(!parseCommand(argc, argv)) {
        cout<<"Error in arguments..\n";
        exit(0);
    }
    
    int *mrna_counts, *batch_pairs;
    double *batch_correlations;
	
	// get memory free on gpu
	float free_m,total_m,used_m;
	size_t free_t,total_t;
	cudaMemGetInfo(&free_t,&total_t);
	free_m =(float)free_t/1048576.0;
	total_m=(float)total_t/1048576.0;
	used_m=total_m-free_m;
	printf ("mem free %f MB, mem total %f MB, mem used %f MB\n", free_m, total_m, used_m);
	
	// load real distributions from cell count matrix
	ifstream infile(path_input_matrix);
	vector<string> rows;
	vector<string> gene_names;
	std::string line;
	
	printf("reading input counts...\n");
	while (getline(infile, line, '\n'))
	{
	    rows.push_back(line); //Get each line of the file as a string
	}
	int num_cells = rows.size() - 1;
	
	// process first row of real counts... first column will be "cell" or empty... this handles them both
	std::size_t last_pos = 0;
	std::size_t pos = rows[0].find(",", last_pos);
	last_pos = pos;
	int num_genes = 0;
	while (pos != std::string::npos){
		pos = rows[0].find(",", last_pos + 1);
		gene_names.push_back(rows[0].substr(last_pos + 1, pos - last_pos - 1));
		last_pos = pos;
		num_genes += 1;
	}
	
	printf("number of genes: %i\n", num_genes);
	printf("number of cells: %i\n", num_cells);
	
	cudaMallocManaged(&mrna_counts, num_genes * num_cells * sizeof(int));
	cudaMallocManaged(&batch_pairs, batch_size * 2 * sizeof(int));
	cudaMallocManaged(&batch_correlations, batch_size * sizeof(double));
	
	printf("processing input matrix...\n");
	int s = rows.size();
	for (int i=1; i<s; ++i){
		// the first thing will be a string cell_id
		last_pos = 0;
		pos = rows[i].find(",", last_pos + 1);
		last_pos = pos;
		
		int i_gene = 0;
		int i_cell = i - 1;
		int count = 0;
		while (pos != std::string::npos){
			pos = rows[i].find(",", last_pos + 1);
			if (rows[i].substr(last_pos + 1, pos - last_pos - 1).empty()) {
				count = 0;
			}
			else {
				count = stoi(rows[i].substr(last_pos + 1, pos - last_pos - 1));
			}
			// initialize cell values
			int i_cell_gene = i_gene * num_cells + i_cell;
			mrna_counts[i_cell_gene] = count;
			last_pos = pos;
			i_gene++;
		}
	}
	
	vector<vector<int>> gene_matrix(2);
	// create parameters combinations
	for (int i_gene = 0; i_gene < num_genes; i_gene++){
		gene_matrix[0].push_back(i_gene);
		gene_matrix[1].push_back(i_gene);
	}
	
	long long int num_gene_pairs = n_choose_r(num_genes, 2);
	
	printf("creating combinations vector...\n");
	vector<vector<int>> gene_pairs_vector = cart_product(gene_matrix);
	
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
				batch_pairs[i_batch_pair * 2] = gene_pairs_vector[i_pair][0];
				batch_pairs[i_batch_pair * 2 + 1] = gene_pairs_vector[i_pair][1];
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
	string path_output_string = string(path_output_dir) + "correlations.csv";
	const char *path_output = path_output_string.c_str();
	
	printf("%s\n", path_output);
	
	outfile = fopen(path_output, "w");//create a file
	fprintf(outfile, "gene_1,gene_2,correlation\n");
	for (long long int i_pair = 0; i_pair < num_gene_pairs; i_pair++){
		fprintf(outfile, "%s,%s,%.16f\n", gene_names[gene_pairs_vector[i_pair][0]].c_str(), gene_names[gene_pairs_vector[i_pair][1]].c_str(), correlations[i_pair]);
	}
	fclose(outfile);
	
	// Free memory
	delete [] correlations;
	gene_pairs_vector.clear();
	cudaFree(mrna_counts);
	cudaFree(batch_pairs);
	cudaFree(batch_correlations);
	return 0;
}




