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
#include <float.h>

using namespace std;

char path_output_dir[200]="path_output_dir";
char path_params[200]="path_params";
char path_counts[200]="path_counts";
double g1_time = 400.0; // 0.25 hours
double s_g2_m_time = 1600.0; // 0.5 hours
double k_deg = 0.0001; // mrna degradation rate
int num_generations = 0;

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
		//printf("argv[%u] = %s\n", i, argv[i]);
        if (strcmp(argv[i], "-o") == 0){
			strcpy(path_output_dir, argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-gt") == 0){
			g1_time=atof(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-st") == 0){
			s_g2_m_time=atof(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-d") == 0){
			k_deg=atof(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-ng") == 0){
			num_generations=atoi(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-p") == 0){
			strcpy(path_params, argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-c") == 0){
			strcpy(path_counts, argv[i+1]);
			i=i+2;
		}
		else{
			return 0;
		}
    }
    return 1;

}

void shuffle(int *array, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
          size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
          int t = array[j];
          array[j] = array[i];
          array[i] = t;
        }
    }
}

__device__
double determine_event(double dt_np, double dt_switch, double dt_express, int *i_event)
{
    if (dt_np <= dt_switch && dt_np <= dt_express){
	    // enter non-permissive
	    *i_event = 0;
	    return dt_np;
    }
    else if (dt_switch <= dt_express){
	    // switch on-off
	    *i_event = 1;
	    return dt_switch;
    }
    else {
	    // transcribe
	    *i_event = 2;
	    return dt_express;
    }
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

__global__
void divide(int *cell_division, int num_cells, int num_genes, int *mrna_count, int *transcriptional_states, int i_generation, curandState* globalState)
{
	//int n, float *x, float *y
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int i_cell_gene = index; i_cell_gene < num_cells * num_genes; i_cell_gene+=stride) {
		
		// x = gene, y = cell
		int i_cell = i_cell_gene / num_genes;
		int i_gene = i_cell_gene % num_genes;
		
		int i_cell_from = -1;
		int i_cell_to = -1;
		
		// find if this cell is in the first half of cell_division
		for (int i_cell_find = 0; i_cell_find<num_cells; i_cell_find++){
			i_cell_from = cell_division[i_cell_find];
			if (i_cell_from == i_cell){
				if (i_cell_find < (int)(num_cells / 2)){
					// cell dividing to is the matching index in the second half of the cell_division
					i_cell_to = cell_division[i_cell_find + (int)(num_cells / 2)];
					break;
				}
				else {
					// this is a cell being divided to
					return;
				}
			}
		}
		
		int i_cell_gene_from = i_cell_from * num_genes + i_gene;
		int i_cell_gene_to = i_cell_to * num_genes + i_gene;
		
		// assign same transcriptional state to new cell
		transcriptional_states[i_cell_gene_to] = transcriptional_states[i_cell_gene_from];
		
		int i_cell_gene_generation_from_this_gen = i_gene + num_genes * i_cell_from + num_genes * num_cells * i_generation;
		
		int i_cell_gene_generation_from_next_gen = i_gene + num_genes * i_cell_from + num_genes * num_cells * (i_generation + 1);
		int i_cell_gene_generation_to_next_gen = i_gene + num_genes * i_cell_to + num_genes * num_cells * (i_generation + 1);
		
		int original_count = mrna_count[i_cell_gene_generation_from_this_gen];
		
		mrna_count[i_cell_gene_generation_from_next_gen] = 0;
		mrna_count[i_cell_gene_generation_to_next_gen] = 0;
		
		for (int i_mrna = 0; i_mrna < original_count; i_mrna++){
			
			double r_divide = generate(globalState, i_cell_gene);
			
			if (r_divide < 0.5){
				// send to new cell
				mrna_count[i_cell_gene_generation_to_next_gen]++;
			} else {
				// keep in this cell
				mrna_count[i_cell_gene_generation_from_next_gen]++;
			}
		}
	}
}

// function to simulate transcriptional bursting
__global__
void simulate(double phase_max_time, int i_phase, int i_generation, int num_cells, int num_genes, int num_params, double *params, int *transcriptional_states, int *mrna_count, curandState* globalState)
{
	//int n, float *x, float *y
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int i_cell_gene = index; i_cell_gene < num_cells * num_genes; i_cell_gene+=stride ) {
		
		// num_genes = width, num_cells = height
		int i_cell = i_cell_gene / num_genes;
		int i_gene = i_cell_gene % num_genes;
		
		int i_cell_gene_generation = i_gene + num_genes * i_cell + num_genes * num_cells * i_generation;
		
		if (i_phase == 1){
			// S/G2/M ... at the start of this, non-permissive states are set to permissive for all genes... except for the ones which are fast k_np (or p_np in other words)
			double r_np = generate(globalState, i_cell_gene);
			// set an initial methylation state (or rather permanent off state) of the gene in a particular cell
			// represents the fact that cells aren't cell-cycle synchronized. some of them will have had more time to get to a completely non-transcriptional state
			if (r_np < params[i_gene * num_params + 3]) {
				transcriptional_states[i_cell_gene] = 2;
			} else {
				transcriptional_states[i_cell_gene] = 0;
			}
		}
		
		double time = 0.0;
		
		while (time < phase_max_time && transcriptional_states[i_cell_gene] < 2) {
				
			double dt_np = -log(generate(globalState, i_cell_gene))/params[i_gene * num_params + 2];
			double dt_switch;
			double dt_express;
			
			if (transcriptional_states[i_cell_gene] == 0){
				// gene is off
				dt_switch = -log(generate(globalState, i_cell_gene))/params[i_gene * num_params + 0];
				dt_express = DBL_MAX;
			} else {
				// gene is on
				dt_switch = -log(generate(globalState, i_cell_gene))/params[i_gene * num_params + 1];
				dt_express = -log(generate(globalState, i_cell_gene))/params[i_gene * num_params + 4];
			}
			
			int i_event;
			double dt = determine_event(dt_np, dt_switch, dt_express, &i_event);
			
			time = time + dt;
			
			if (time < phase_max_time){
				
				// before adding another mrna possibly, figure out how many mrnas degrade over the time it takes to reach the next step
				int num_degraded_mrnas = 0;
				for (int i_mrna = 0; i_mrna < mrna_count[i_cell_gene_generation]; i_mrna++){
					double dt_degradation = -log(generate(globalState, i_cell_gene))/(params[i_gene * num_params + 5] * (double)mrna_count[i_cell_gene]); // incorporate current concentration for first order 
					if (dt_degradation <= dt){
						num_degraded_mrnas++;
					}
				}
				
				mrna_count[i_cell_gene_generation] = mrna_count[i_cell_gene_generation] - num_degraded_mrnas;
				
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
				}
				else if (i_event == 2){
					// transcribe
					mrna_count[i_cell_gene_generation]++;
				}
			}
		}
	}
}

// nvcc /home/data/nlaszik/cuda_simulation/code/cuda/cell_cycle_gillespie_all_parallel.cu -o /home/data/nlaszik/cuda_simulation/code/cuda/build/cell_cycle_gillespie_all_parallel -lcurand -lineinfo

// SRP299892
// /home/data/nlaszik/cuda_simulation/code/cuda/build/cell_cycle_gillespie_all_parallel -ng 5 -gt 400 -st 1600 -d 0.00005 -c /home/data/Shared/shared_datasets/sc-rna-seq/SRP299892/seurat/transcript_counts/srr13336770_transcript_counts.filtered.norm.csv -p /home/data/nlaszik/cuda_simulation/output/SRP299892/SRR13336770_norm_gillespie/initial_parameters.csv -o /home/data/nlaszik/cuda_simulation/output/SRP299892/SRR13336770_norm_gillespie/

int main(int argc, char** argv)
{
	
	if (__cplusplus == 202002L) std::cout << "C++20\n";
	else if (__cplusplus == 201703L) std::cout << "C++17\n";
    else if (__cplusplus == 201402L) std::cout << "C++14\n";
    else if (__cplusplus == 201103L) std::cout << "C++11\n";
    else if (__cplusplus == 199711L) std::cout << "C++98\n";
    else std::cout << "pre-standard C++\n";
	
	if(!parseCommand(argc, argv)) {
        cout<<"Error in arguments..\n";
        exit(0);
    }
    
    // display input arguments
    printf("number of generations: %i\n", num_generations);
	
	// 2d array is really 1d array... easier to manage memory this way
	int *transcriptional_states, *mrna_count, *cell_division;
	double *params;
	int num_params = 6;
	
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
	
	cudaMallocManaged(&params, num_genes * num_params * sizeof(double));
	
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
		
		int i_param = 0;
		double parameter = 0.0;
		while (pos != std::string::npos){
			pos = rows_params[i].find(",", last_pos + 1);
			if (rows_params[i].substr(last_pos + 1, pos - last_pos - 1).empty()) {
				parameter = 0.0;
			}
			else {
				parameter = stod(rows_params[i].substr(last_pos + 1, pos - last_pos - 1));
				int i_param_gene = i_gene * num_params + i_param;
				params[i_param_gene] = parameter;
				i_param++;
			}
			// initialize param values
			last_pos = pos;
		}
		// add placeholder mrna degradation rate
		int i_param_gene = i_gene * num_params + i_param;
		params[i_param_gene] = k_deg;
		
		/*
		for (int i_param_check = 0; i_param_check < num_params; i_param_check++){
			int i_param_gene_check = i_gene * num_params + i_param_check;
			printf("%f,", params[i_param_gene_check]);
		}
		printf("\n");
		*/
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
	int num_cells = rows.size() - 1;
	
	// need an even number of cells
	if (num_cells % 2 != 0){
		num_cells = num_cells - 1;
	}
	
	cudaMallocManaged(&cell_division, num_cells * sizeof(int));
	cudaMallocManaged(&transcriptional_states, num_genes * num_cells * sizeof(int));
	cudaMallocManaged(&mrna_count, num_genes * num_cells * num_generations * sizeof(int));
	
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
	
	printf("number of cells in counts: %i\n", num_cells);
	printf("number of genes in counts: %i\n", num_genes_counts);
	
	if (num_genes_counts != num_genes){
		printf("number of genes in counts, %i, does not match with number of genes in params, %i, please fix...\n", num_genes_counts, num_genes);
		exit(0);
	}
	
	double cell_cycle_progress_at_s = g1_time / (g1_time + s_g2_m_time);
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
			if (i_cell < num_cells){
				int i_cell_gene_generation = i_gene + num_genes * i_cell + num_genes * num_cells * 0;
				//mrna_count[i_cell_gene_generation] = count; // scale the count by the progress through the cell cycle
				mrna_count[i_cell_gene_generation] = (int)((double)count * cell_cycle_progress_at_s); // scale the count by the progress through the cell cycle
			}
			last_pos = pos;
			i_gene++;
		}
	}
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator (seed);
	
	// setup and allocate memory for curand
	int N = num_genes * num_cells;
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
	
	cudaEventRecord(start);
	
	// a generation defined as G1->M
	for (int i_generation = 0; i_generation < num_generations; i_generation++){
		
		for (int i_phase = 0; i_phase < 2; i_phase++){
			
			if (i_generation == 0 && i_phase == 0){
				// if this is the first generation, we want to start at S-phase so that our np's are set appropriately according to p_np
				continue;
			}
			
			double phase_max_time;
			if (i_phase == 1){
				phase_max_time = s_g2_m_time;
			}
			else {
				phase_max_time = g1_time;
			}
			
			printf("Simulating generation %i, phase %i:\n", i_generation, i_phase);
			
			simulate<<<numBlocks, blockSize>>>(phase_max_time, i_phase, i_generation, num_cells, num_genes, num_params, params, transcriptional_states, mrna_count, devStates);
			cudaEventRecord(stop);
			cudaDeviceSynchronize();
			
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("Elapsed seconds: %f\n", milliseconds/1000);
			
		}
		
		// divide
		if (i_generation < num_generations - 1){
			// randomly select 50% of cells for dividing, will throw out the rest
			// shuffle two arrays containing all cell indices
			// first half of cell_division be source, second half will be target
			for (int i_cell = 0; i_cell < num_cells; i_cell++) {
				// reset array
				cell_division[i_cell] = i_cell;
			}
			shuffle(cell_division, num_cells);
			
			// distribute mrna randomly as cells divide
			divide<<<numBlocks, blockSize>>>(cell_division, num_cells, num_genes, mrna_count, transcriptional_states, i_generation, devStates);
			
			cudaEventRecord(stop);
			cudaDeviceSynchronize();
			
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("Elapsed seconds for division: %f\n", milliseconds/1000);
		}
	}
	
	//output file to csv
	
	FILE *outfile;
	string path_output_string = string(path_output_dir) + "cell_cycle_counts_" + to_string((int)(s_g2_m_time + g1_time)) + "_" + to_string(num_generations) + "_" + to_string(k_deg) + ".csv";
	const char *path_output = path_output_string.c_str();
	
	printf("%s\n", path_output);
	
	outfile = fopen(path_output, "w");//create a file
	for (int i_gene = 0; i_gene < num_genes; i_gene++){
		fprintf(outfile, "%s,", gene_names[i_gene].c_str());
		for (int i_generation = 0; i_generation < num_generations; i_generation++){
		    for (int i_cell = 0; i_cell < num_cells; i_cell++){
			    int i_cell_gene_generation = i_gene + num_genes * i_cell + num_genes * num_cells * i_generation;
		        fprintf(outfile, "%i,", mrna_count[i_cell_gene_generation]);
		    }
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




