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

char path_output_dir[200]="path_output_dir";
char path_params[200]="path_params";
int num_timesteps = 0;
int max_count = 0;
int num_cells = 0;
double max_time = 3600.0; // 1 hour

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

float randomFloat(float min, float max) {
   return ((max - min) * ((float)rand() / RAND_MAX)) + min;
}

int parseCommand(int argc, char **argv) {
    for(int i=1;i<argc;) {
		printf("argv[%u] = %s\n", i, argv[i]);
        if (strcmp(argv[i], "-o") == 0){
			strcpy(path_output_dir, argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-nt") == 0){
			num_timesteps=atoi(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-nc") == 0){
			num_cells=atoi(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-p") == 0){
			strcpy(path_params, argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-mc") == 0){
			max_count=atoi(argv[i+1]);
			i=i+2;
		}
		else{
			return 0;
		}
    }
    return 1;

}

__device__
float generate(curandState* globalState, int ind)
{
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__global__
void setup_kernel(curandState * state, unsigned long seed, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) curand_init ( seed, id, 0, &state[id] );
}

// function to simulate transcriptional bursting
__global__
void simulate(double time_to_replication, double time_to_division, int num_cells, int num_genes, int num_params, double *params, int *transcriptional_states, int *mrna_count, curandState* globalState)
{
	//int n, float *x, float *y
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int i_cell = index; i_cell < num_cells_current_phase; i_cell+=stride) {
		
		float current_phase_max_time = 0.0;
		
		// need some global of how many cells there are currently
		if (cell_cycle_states[i_cell] == 0){
			// cell is not alive yet
		}
		else if (cell_cycle_states[i_cell] == 1){
			// G1 ... at the end of this, non-permissive states are reset to permissive for all genes
			current_phase_max_time = 1000.0;
		}
		else if (cell_cycle_states[i_cell] == 2){
			// S/G2/M ... at the end of this, mrna counts are halved
			current_phase_max_time = 5000.0;
		}
		
		int i_cell_gene;
		
		for (int i_gene = 0; i_gene < num_genes; i_gene++) {
			
			i_cell_gene = i_cell * num_genes + i_gene;
		
			double r_np = generate(globalState, i_cell);
			// set an initial methylation state (or rather permanent off state) of the gene in a particular cell
			// represents the fact that cells aren't cell-cycle synchronized. some of them will have had more time to get to a completely non-transcriptional state
			if (r_np < params[i_gene * num_params + 3]) {
				transcriptional_states[i_cell_gene] = 2;
			}
			else {
				
				double time = 0.0;
				int iteration = 0;
				
				transcriptional_states[i_cell_gene] = 0;
				
				// TODO : end when current cell cycle phase is over
				while (time < time_to_division && transcriptional_states[i_cell_gene] < 2) {
						
					double dt_np = -log(generate(globalState, i_cell))/params[i_gene * num_params + 2];
					double dt_switch;
					double dt_express;
					
					if (transcriptional_states[i_cell_param_combination] == 0){
						// gene is off
						dt_switch = -log(generate(globalState, i_cell))/params[i_gene * num_params + 0];
						dt_express = DBL_MAX;
					} else {
						// gene is on
						dt_switch = -log(generate(globalState, i_cell))/params[i_gene * num_params + 1];
						dt_express = -log(generate(globalState, i_cell))/params[i_gene * num_params + 4];
					}
					
					int i_event;
					double dt = determine_event(dt_np, dt_switch, dt_express, &i_event);
					
					time = time + dt;
					iteration++;
					
					if (time < current_phase_max_time){
						if (i_event == 0){
						// np
							transcriptional_states[i_cell_gene] = 2;
						}
						else if (i_event == 1){
							// switch
							if (transcriptional_states[i_cell_gene] == 1){
								transcriptional_states[i_cell_gene] = 0;
							} else {
								transcriptional_states[i_cell_gene] = 1;
							}
						} else {
							// transcribe
							mrna_count[i_cell_gene]++;
						}
					}
				}
				
				// need some global of how many cells there are currently
				if (cell_cycle_states[i_cell] == 1){
					// G1 ... at the end of this, non-permissive states are reset to permissive for all genes
					for (int i_gene = 0; i_gene < num_genes; i_gene++){
						i_cell_gene = i_cell * num_genes + i_gene;
						transcriptional_states[i_cell_gene] = 0;
					}
					cell_cycle_states[i_cell] == 2;
				}
				else if (cell_cycle_states[i_cell] == 2){
					// S/G2/M ... at the end of this, mrna counts are halved between original cell and some other cell
					cell_cycle_states[i_cell] == 1;
				}
			}
		}
	}
}

// nvcc /home/data/nlaszik/cuda_simulation/code/cuda/simulate_with_parameters_genes.cu -o /home/data/nlaszik/cuda_simulation/code/cuda/build/simulate_with_parameters -lcurand

// SRP215251
// /home/data/nlaszik/cuda_simulation/code/cuda/build/simulate_with_parameters -nt 5000 -mc 400 -p /home/data/nlaszik/cuda_simulation/output/SRP215251/DKO/initial_parameters.csv -o /home/data/nlaszik/cuda_simulation/output/SRP215251/DKO/

// /home/data/nlaszik/cuda_simulation/code/cuda/build/simulate_with_parameters -nt 5000 -mc 400 -p /home/data/nlaszik/cuda_simulation/output/SRP215251/WT/initial_parameters.csv -o /home/data/nlaszik/cuda_simulation/output/SRP215251/WT/

// SRP313343
// /home/data/nlaszik/cuda_simulation/code/cuda/build/simulate_with_parameters -nt 5000 -mc 400 -p /home/data/nlaszik/cuda_simulation/output/SRP313343/SRR14139729/initial_parameters.csv -o /home/data/nlaszik/cuda_simulation/output/SRP313343/SRR14139729/

// /home/data/nlaszik/cuda_simulation/code/cuda/build/simulate_with_parameters -nt 5000 -mc 400 -p /home/data/nlaszik/cuda_simulation/output/SRP313343/SRR14139730/initial_parameters.csv -o /home/data/nlaszik/cuda_simulation/output/SRP313343/SRR14139730/

// SRP299892
// /home/data/nlaszik/cuda_simulation/code/cuda/build/simulate_with_parameters -nt 5000 -mc 400 -p /home/data/nlaszik/cuda_simulation/output/SRP299892/SRR13336770/initial_parameters.csv -o /home/data/nlaszik/cuda_simulation/output/SRP299892/SRR13336770/

int main(int argc, char** argv)
{
	
	if(!parseCommand(argc, argv)) {
        cout<<"Error in arguments..\n";
        exit(0);
    }
	
	// 2d array is really 1d array... easier to manage memory this way
	int *transcriptional_states, *mrna_count, *params;
	int num_params = 5;
	
	// get params so we can write them in for genes
	printf("reading parameters file...\n");
	ifstream paramsfile(path_params);
	vector<string> rows_params;
	std::string line_params;
	vector<string> gene_names;
	
	while (getline(paramsfile, line_params, '\n'))
	{
	    rows_params.push_back(line_params); //Get each line of the file as a string
	}
	
	int num_genes = rows_params.size() - 1;
	
	cudaMallocManaged(&transcriptional_states, num_genes * max_cells * sizeof(int));
	cudaMallocManaged(&mrna_count, num_genes * max_cells * sizeof(int));
	cudaMallocManaged(&params, num_genes * num_params * sizeof(int));
	
	printf("number of genes in params: %i\n", num_genes);
	
	std::size_t last_pos = 0;
	std::size_t pos = 0;
	last_pos = pos;
	
	for (int i=1; i<rows_params.size(); ++i){
		// the first thing will be a string gene_id
		int i_gene = i - 1;
		
		last_pos = -1;
		pos = rows_params[i].find(",", last_pos + 1);
		gene_names.push_back(rows_params[i].substr(last_pos + 1, pos - last_pos - 1));
		last_pos = pos;
		
		printf("%s\n", gene_names[i_gene].c_str());
		
		int i_param = 0;
		int parameter = 0;
		while (pos != std::string::npos){
			pos = rows_params[i].find(",", last_pos + 1);
			if (rows_params[i].substr(last_pos + 1, pos - last_pos - 1).empty()) {
				parameter = 0;
			}
			else {
				parameter = stoi(rows_params[i].substr(last_pos + 1, pos - last_pos - 1));
				int i_param_gene = i_gene * num_params + i_param;
				params[i_param_gene] = parameter;
				i_param++;
			}
			// initialize param values
			last_pos = pos;
		}
	}
	
	ifstream infile(path_counts);
	vector<string> rows;
	vector<string> gene_names_counts;
	std::string line;
	
	printf("reading input counts...\n");
	while (getline(infile, line, '\n'))
	{
	    rows.push_back(line); //Get each line of the file as a string
	}
	int num_cells_counts = rows.size() - 1;
	
	int num_genes_counts = 0;
	// process first row of real counts... first column will be "cell" or empty... this handles them both
	last_pos = 0;
	pos = rows[0].find(",", last_pos);
	last_pos = pos;
	while (pos != std::string::npos){
		pos = rows[0].find(",", last_pos + 1);
		gene_names_counts.push_back(rows[0].substr(last_pos + 1, pos - last_pos - 1));
		last_pos = pos;
		num_genes_counts += 1;
	}
	
	printf("number of cells in counts: %i\n", num_cells_counts);
	printf("number of genes in counts: %i\n", num_genes_counts);
	
	if (num_genes_counts != num_genes){
		printf("number of genes in counts does not match with number of genes in params, please fix...\n");
		exit(1);
	}
	
	int *real_mrna_means = new int[num_genes_counts];
	
	printf("getting mean counts...\n");
	// init real distributions
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
			//int i_cell_gene = i_cell * num_genes_counts + i_gene;
			real_mrna_means[i_gene] += count;
			last_pos = pos;
			i_gene++;
		}
	}
	
	for (int i_gene = 0; i_gene<num_genes_counts; i_gene++){
		
	}
	
	printf("number of genes: %i\n", num_genes);
	printf("number of cells: %i\n", num_cells);
	printf("max count: %i\n", max_count);
	printf("timesteps: %i\n", num_timesteps);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator (seed);
	
	// setup and allocate memory for curand
	int N = num_genes * max_cells;
	int *y, *d_y;
	y = (int*)malloc(N*sizeof(int));
	
	cudaMalloc(&d_y, N * sizeof(int));
	cudaMemcpy(d_y, y, N * sizeof(int), cudaMemcpyHostToDevice);
	
	curandState* devStates;
	cudaMalloc (&devStates, N * sizeof(curandState));
	
	// Run kernel on the GPU
	int blockSize = 32;
	int numBlocks = (N + blockSize - 1) / blockSize;
	
	setup_kernel<<<numBlocks, blockSize>>>(devStates, seed, N);
	
	// a generation defined as G1->M
	for (int i_generation = 0; i_generation < num_generations; i_generation++){
		cudaEventRecord(start);
		simulate<<<numBlocks, blockSize>>>(num_timesteps, num_cells, num_genes, num_params, max_count, params, transcriptional_states, mrna_count, devStates);
		cudaEventRecord(stop);
		cudaDeviceSynchronize();
	}
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	printf("Elapsed seconds: %f\n", milliseconds/1000);
	
	//output file to csv
	
	FILE *outfile;
	string path_output_string = string(path_output_dir) + "initial_counts.csv";
	const char *path_output = path_output_string.c_str();
	
	printf("%s\n", path_output);
	
	outfile = fopen(path_output, "w");//create a file
	for (int i_gene = 0; i_gene < num_genes; i_gene++){
		fprintf(outfile, "%s,", gene_names[i_gene].c_str());
	    for (int i_cell = 0; i_cell < num_cells; i_cell++){
	        fprintf(outfile, "%i,", mrna_count[i_cell * num_genes + i_gene]);
	    }
	    fprintf(outfile,"\n");
	}
	fclose(outfile);
	
	// Free memory
	cudaFree(transcriptional_states);
	cudaFree(params);
	cudaFree(mrna_count);
	cudaFree(devStates);
	cudaFree(d_y);
	return 0;
}




