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

#define MIN(a,b) (((a)<(b))?(a):(b))

using namespace std;

char path_output_dir[200]="path_output_dir";
char path_input_matrix[200]="path_input_matrix";
char path_params[200]="path_params";
int num_timesteps = 5000;
int max_count = 400;
int step = 100;
double h = 4.0; // bandwidth for kde
int num_iterations = 1000;

__device__
auto k_gpu(double val)
{
    const double p=1.0 / std::sqrt( 2.0 * M_PI);
    return p * std::exp(- 0.5 * (val*val));
}

auto k(double val)
{
    static const double p=1.0 / std::sqrt( 2.0 * M_PI);
    return p * std::exp(- 0.5 * (val*val));
}

__device__
auto generate_kde_gpu(double *distributions, int *mrna_counts, int max_count, double h, int batch_size, int i_param_combination, int num_cells)
{
	const double x_0 = 0.0;
	const int Nx = max_count;
	const double x_limit = (double)max_count;
	const double p = 1.0 / (h * max_count);
	const double hx = (x_limit - x_0)/(Nx - 1);
	
	for(int i_x = 0; i_x < Nx; ++i_x)
	{
		int i_dist = i_param_combination * max_count + i_x;
	    double x = x_0 + i_x * hx;
	    double sum = 0;
	    for (int i_cell = 0; i_cell < num_cells; i_cell++) {
		    int i_cell_param_combination = i_cell * batch_size + i_param_combination;
			//printf("filling distribution for cell %i, param combination %i, total index %i\n", i_cell, i_param_combination, i_cell_param_combination);
			//if (i_param_combination == 0) printf("%i,", mrna_counts[i_cell_param_combination]);
			sum += k_gpu((x - (double)mrna_counts[i_cell_param_combination]) / h);
		}
	    distributions[i_dist] = p * sum * 10;
	}
};

auto generate_kde(double *distributions, int *mrna_counts, int max_count, double h, int num_genes, int i_gene, int num_cells)
{
	const double x_0 = 0.0;
	const int Nx = max_count;
	const double x_limit = (double)max_count;
	const double p = 1.0 / (h * max_count);
	const double hx = (x_limit - x_0)/(Nx - 1);
	
	for(int i_x = 0; i_x < Nx; ++i_x)
	{
		int i_dist = i_gene * max_count + i_x;
	    double x = x_0 + i_x * hx;
	    double sum = 0;
	    for (int i_cell = 0; i_cell < num_cells; i_cell++) {
			int i_cell_gene = i_cell * num_genes + i_gene;
			sum += k((x - (double)mrna_counts[i_cell_gene]) / h);
		}
	    distributions[i_dist] = p * sum * 10;
	    //printf("%f,", p * sum * 10);
	}
	//printf("\n");
};

auto kde(double x_limit, std::vector<double> values, double h)
{
	const double x_0 = 0.0;
	const int len = values.size();
	//const int Nx = 100;
	const int Nx = x_limit;
	const double p = 1.0 / (h * len);
	const double hx = (x_limit - x_0)/(Nx - 1);
	
	std::vector<double> output;
	
	for(int i_x = 0; i_x < Nx; ++i_x)
	{
	    double x = x_0 + i_x * hx;
	    double sum = 0;
	    for(int i = 0; i < len; i++)
	        sum += k((x - values[i]) / h);
	    output.push_back(p * sum);
	}
	return output;
};

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
		else if (strcmp(argv[i], "-i") == 0){
			strcpy(path_input_matrix, argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-p") == 0){
			strcpy(path_params, argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-nt") == 0){
			num_timesteps=atoi(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-ni") == 0){
			num_iterations=atoi(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-s") == 0){
			step=atoi(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-h") == 0){
			h=atof(argv[i+1]);
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

__device__
float log_likelihood(double *real_distributions, int *mrna_count, int i_gene, int num_cells, int num_genes, int max_count) {
	double likelihood_product = 0.0;
	for (int i_cell = 0; i_cell < num_cells; i_cell++){
		int i_cell_gene = i_cell * num_genes + i_gene;
		int i_dist = i_gene * max_count + mrna_count[i_cell_gene];
		//printf("i_dist: %i\n", i_dist);
		
		//likelihood_product *= real_distributions[i_dist];
		
		// if statement to discount -inf and inf... not sure if this is good
		double single_value_likelihood = log(real_distributions[i_dist]);
		if (single_value_likelihood != ((double)-1.0/(double)0.0) && single_value_likelihood != ((double)1.0/(double)0.0)){
			likelihood_product += single_value_likelihood;
		}
	}
	return likelihood_product;
}

__global__
void mcmc(int num_timesteps, int num_cells, int num_genes, int num_iterations, int *step_sizes, const int num_params, int max_count, int *transcriptional_states, int *proposed_mrna_count, int *current_mrna_count, double *real_distributions, int *best_params, curandState* globalState){
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int i_gene = index; i_gene < num_genes; i_gene+=stride) {
		
		// reset counts
		for (int i_cell = 0; i_cell < num_cells; i_cell++){
			int i_cell_gene = i_cell * num_genes + i_gene;
			proposed_mrna_count[i_cell_gene] = 0;
		}
		
		for (int iteration = 0; iteration < num_iterations; iteration++) {
			
			// only change params on second iteration, first iteration fills input counts
			int direction;
			if (iteration == 0){
				direction = 0;
			} else {
				direction = (int)(ceil(generate(globalState, i_gene)*(1 + 1)) - 1) * 2 - 1; // -1 or 1 for direction
			}
			
			// choose param to change
			int i_param_to_change = (int)(ceil(generate(globalState, i_gene)*(num_params)) - 1); // choose a parameter at random to alter
			int previous_param_value = best_params[i_gene * num_params + i_param_to_change];
			int new_param_value = best_params[i_gene * num_params + i_param_to_change] + direction * step_sizes[i_param_to_change];
			
			// only change to a param value that makes sense... keep picking a random param and direction until conditions satisfied
			// can't go past limits, and p_on and p_tx can never be 0
			if (iteration > 0){
				while (new_param_value < 0 || new_param_value > 1000 || ((i_param_to_change == 0 || i_param_to_change == 4) && new_param_value == 0) || (i_param_to_change == 1 && new_param_value == 1000)){
					direction = (int)(ceil(generate(globalState, i_gene)*(1 + 1)) - 1) * 2 - 1; // -1 or 1 for direction
					i_param_to_change = (int)(ceil(generate(globalState, i_gene)*(num_params)) - 1); // choose a parameter at random to alter... 0 to num_params
					previous_param_value = best_params[i_gene * num_params + i_param_to_change];
					new_param_value = best_params[i_gene * num_params + i_param_to_change] + direction * step_sizes[i_param_to_change];
				}
			}
			
			best_params[i_gene * num_params + i_param_to_change] = new_param_value;
			
			//printf("gene:%i, iteration:%i, old param:%i, new_param:%i\n", i_gene, iteration, previous_param_value, new_param_value);
			
			for (int i_cell = 0; i_cell < num_cells; i_cell++) {
				
				int i_cell_gene = i_cell * num_genes + i_gene;
				
				//printf("trying to simulate gene: %i, cell: %i, total index: %i\n", i_gene, i_cell, i_cell_gene);
				
				int r_age = (int)(ceil(generate(globalState, i_gene)*(1000 + 1)) - 1);
				//printf("gene: %i, age: %i\n", i_gene, r_age);
				// set an initial methylation state (or rather permanent off state) of the gene in a particular cell
				// represents the fact that cells aren't cell-cycle synchronized. some of them will have had more time to get to a completely non-transcriptional state
				if (r_age < best_params[i_gene * num_params + 3]) {
					transcriptional_states[i_cell_gene] = 2;
				}
				else {
					
					transcriptional_states[i_cell_gene] = 0;
					int timestep = 0;
					
					while (timestep < num_timesteps && proposed_mrna_count[i_cell_gene] < max_count) {
						
						// only 1 thing can happen at a time
						int r_methylation = (int)(ceil(generate(globalState, i_gene)*(num_timesteps*100 + 1)) - 1); // random uniform number, around 1% of genes will methylate over timecourse if p=1
						
						// assess transcriptional state of gene
						//printf("assessing state %i, cell %i, total index %i\n", i_gene, i_cell, i_cell_gene);
						if (transcriptional_states[i_cell_gene] < 2){
							if (r_methylation < best_params[i_gene * num_params + 2]) {
								transcriptional_states[i_cell_gene] = 2;
							}
						}
						
						if (transcriptional_states[i_cell_gene] < 2) {
							//printf("assessing switch %i, cell %i, total index %i\n", i_gene, i_cell, i_cell_gene);
							int r_switch = (int)(ceil(generate(globalState, i_gene)*(1000 + 1)) - 1); //random uniform number, 0-1000
							if (transcriptional_states[i_cell_gene] == 0) {
								if (r_switch < best_params[i_gene * num_params + 0]) {
									transcriptional_states[i_cell_gene] = 1;
								}
							} else {
								if (r_switch < best_params[i_gene * num_params + 1]) {
									transcriptional_states[i_cell_gene] = 0;
								} else {
									//printf("assessing transcription %i, cell %i, total index %i\n", i_gene, i_cell, i_cell_gene);
									int r_transcription = (int)(ceil(generate(globalState, i_gene)*(1000 + 1)) - 1); //random uniform number, 0-1000
									if (r_transcription < best_params[i_gene * num_params + 4]) {
										//printf("transcription occurred\n");
										proposed_mrna_count[i_cell_gene]++;
									}
								}
							}
						}
						timestep++;
					}
				}
				//printf("simulated for gene %i, cell %i, total index %i, mrna count: %i\n", i_gene, i_cell, i_cell_gene, proposed_mrna_count[i_cell_gene]);
			}
			
			//printf("finished simulating for gene %i\n", i_gene);
			
			// boltzmann test... need to figure out how to alter this based on the iteration... needs to decrease to zero since represents energy
			// the change over iterations should alter the floor of the random number. the criteria for acceptance when failing should get stricter and stricter
			// floor should be close to 0 for most of the time, then slowly grow towards 1
			// starting at 11.8 since we have already gotten close to true value, we want very little energy and mostly likelihood descent
			
			// goes above 1.0 at around 260 its with 11.8 intercept
			// goes above 1.0 at around 600 its with 11.5 intercept
			// starts at 1.0 with 12.0 intercept
			// without min(), the acceptance ratio must keep improving... with 11.8 intercept, it needs to be about a 20% improvement by the end
			
			//double floor = (-log2((double)2048.0 - (double)iteration) + (double)12.0);
			//double test_probability = min((double)1.0, ((double)generate(globalState, i_gene) * ((double)1.0 - floor)) + floor);
			//double test_probability = (double)generate(globalState, i_gene) * ((double)1.0 - floor) + floor;
			//double test_probability = (double)generate(globalState, i_gene) * ((double)1.4 - floor) + floor;
			
			// floor by itself... starts with a 20% improvement and goes down to 0% improvement by 300 its. maybe don't need random noise at this point and just deterministic descent
			double floor = (-log2((double)2048.0 + (double)iteration) + (double)12.2);
			
			//printf("gene: %i, prob: %f\n", i_gene, test_probability);
			
			// this is the maximum likelihood
			double proposed_likelihood = log_likelihood(real_distributions, proposed_mrna_count, i_gene, num_cells, num_genes, max_count);
			//printf("gene: %i, proposed likelihood: %f\n", i_gene, proposed_likelihood);
			double current_likelihood = log_likelihood(real_distributions, current_mrna_count, i_gene, num_cells, num_genes, max_count);
			//printf("gene: %i, current likelihood: %f\n", i_gene, current_likelihood);
			
			//double acceptance_ratio = min((double)1.0, exp(proposed_likelihood - current_likelihood));
			double acceptance_ratio = (double)exp(proposed_likelihood - current_likelihood);
			
			if(acceptance_ratio < floor) { //Reject
				//printf("gene %i iteration %i: failed in changing param %i from %i to %i\n", i_gene, iteration, i_param_to_change, previous_param_value, new_param_value);
				best_params[i_gene * num_params + i_param_to_change] = previous_param_value;
				for (int i_cell = 0; i_cell < num_cells; i_cell++){
					int i_cell_gene = i_cell * num_genes + i_gene;
					proposed_mrna_count[i_cell_gene] = current_mrna_count[i_cell_gene];
				}
	        } else {
		        printf("gene %i iteration %i ratio %f prob %f: succeeded in changing param %i from %i to %i\n", i_gene, iteration, acceptance_ratio, floor, i_param_to_change, previous_param_value, new_param_value);
		        for (int i_cell = 0; i_cell < num_cells; i_cell++){
					int i_cell_gene = i_cell * num_genes + i_gene;
					current_mrna_count[i_cell_gene] = proposed_mrna_count[i_cell_gene];
				}
	        }
		}
	}
}

// 3D to 1D and reverse
// x = i % width;
// y = (i / width)%height;
// z = i / (width*height);
//i = x + width*y + width*height*z;

// nvcc /home/data/nlaszik/cuda_simulation/code/cuda/mcmc.cu -o /home/data/nlaszik/cuda_simulation/code/cuda/build/mcmc -lcurand -lineinfo

// SRP299892
// smaller batch size due to higher # of cells?
// /home/data/nlaszik/cuda_simulation/code/cuda/build/mcmc -nt 5000 -mc 400 -s 20 -h 4.0 -ni 500 -i /home/data/Shared/shared_datasets/sc-rna-seq/SRP299892/seurat/transcript_counts/srr13336770_transcript_counts.filtered.norm.csv -o /home/data/nlaszik/cuda_simulation/output/SRP299892/SRR13336770_norm_run_00/ -p /home/data/nlaszik/cuda_simulation/output/SRP299892/SRR13336770_norm_run_00/initial_parameters.csv

int main(int argc, char** argv)
{
	
	if(!parseCommand(argc, argv)) {
	    cout<<"Error in arguments..\n";
	    exit(0);
	}
	
	printf("max count: %i\n", max_count);
	printf("num timesteps: %i\n", num_timesteps);
	printf("step size: %i\n", step);
	printf("h: %f\n", h);
	printf("path_params: %s\n", path_params);
	
	float free_m,total_m,used_m;

	size_t free_t,total_t;
	
	cudaMemGetInfo(&free_t,&total_t);
	
	free_m =(float)free_t/1048576.0;
	
	total_m=(float)total_t/1048576.0;
	
	used_m=total_m-free_m;
	
	printf ("mem free %f MB, mem total %f MB, mem used %f MB\n", free_m, total_m, used_m);
	
	// load real distributions from cell count matrix
    // also make initial guesses for parameters
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
	
	int num_genes = 0;
	// process first row of real counts... first column will be "cell" or empty... this handles them both
	std::size_t last_pos = 0;
	std::size_t pos = rows[0].find(",", last_pos);
	last_pos = pos;
	while (pos != std::string::npos){
		pos = rows[0].find(",", last_pos + 1);
		gene_names.push_back(rows[0].substr(last_pos + 1, pos - last_pos - 1));
		last_pos = pos;
		num_genes += 1;
	}
	
	printf("number of cells: %i\n", num_cells);
	printf("number of genes: %i\n", num_genes);
	
	int *best_params;
	int *real_mrna_count = new int[num_cells * num_genes];
	double *real_distributions;
	
	cudaMallocManaged(&real_distributions, num_genes * max_count * sizeof(double));
	
	printf("generating real count kdes...\n");
	// init real distributions
	for (int i_fill = 0; i_fill < num_genes * max_count; i_fill++) {
		real_distributions[i_fill] = 0.0;
	}
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
			int i_cell_gene = i_cell * num_genes + i_gene;
			real_mrna_count[i_cell_gene] = count;
			last_pos = pos;
			i_gene++;
		}
	}
	for (int i_gene = 0; i_gene < num_genes; i_gene++) {
		// interpolate distribution counts
		generate_kde(real_distributions, real_mrna_count, max_count, h, num_genes, i_gene, num_cells);
	}
	
	int *transcriptional_states, *proposed_mrna_count, *current_mrna_count, *step_sizes;
	
	cudaMallocManaged(&transcriptional_states, num_genes * num_cells * sizeof(int));
	cudaMallocManaged(&proposed_mrna_count, num_genes * num_cells * sizeof(int));
	cudaMallocManaged(&current_mrna_count, num_genes * num_cells * sizeof(int));
	
	// fill in starting current count
	for (int i_fill = 0; i_fill < num_genes * num_cells; i_fill++) {
		// interpolate distribution counts
		current_mrna_count[i_fill] = 0;
	}
	
	// get params so we can write them in for genes
	printf("reading parameters file...\n");
	ifstream paramsfile(path_params);
	vector<string> rows_params;
	std::string line_params;
	
	while (getline(paramsfile, line_params, '\n'))
	{
	    rows_params.push_back(line_params); //Get each line of the file as a string
	}
	
	int num_params = 0;
	
	last_pos = 0;
	pos = rows_params[0].find(",", last_pos);
	last_pos = pos;
	
	while (pos != std::string::npos){
		pos = rows_params[0].find(",", last_pos + 1);
		if (rows_params[0].substr(last_pos + 1, pos - last_pos - 1).empty()) {
			int holder = 0;
		}
		else {
			num_params++;
		}
		// initialize param values
		last_pos = pos;
	}
	
	printf("number of parameters: %i\n", num_params);
	
	cudaMallocManaged(&best_params, num_genes * num_params * sizeof(int));
	
	int num_genes_in_params = 0;
	for (int i=1; i<rows_params.size(); ++i){
		// the first thing will be a string cell_id
		last_pos = -1;
		pos = 0;
		int i_gene = i - 1;
		int i_param = 0;
		int parameter = 0;
		
		//printf("%s\n", rows_params[i].c_str());
		
		while (pos != std::string::npos){
			
			pos = rows_params[i].find(",", last_pos + 1);
			
			if (rows_params[i].substr(last_pos + 1, pos - last_pos - 1).empty()) {
				parameter = 0;
			}
			else {
				// skip gene name, only grab param values
				if (i_param > 0){
					parameter = stoi(rows_params[i].substr(last_pos + 1, pos - last_pos - 1));
					int i_param_gene = i_gene * num_params + (i_param - 1);
					best_params[i_param_gene] = parameter;
				}
				i_param++;
			}
			// initialize param values
			last_pos = pos;
		}
		num_genes_in_params++;
	}
	
	printf("num genes in params file: %i\n", num_genes_in_params);
	
	cudaMallocManaged(&step_sizes, num_params * sizeof(int));
	
	// SET STEP SIZES
	for (int i=0; i<num_params; ++i){
		step_sizes[i] = step;
	}
	step_sizes[0] = int(step/4);
	
	// set up to fit on gpu
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator (seed);
	
	// setup and allocate memory for curand
	int N = num_genes;
	int *y, *d_y;
	y = (int*)malloc(N * sizeof(int));
	
	cudaMalloc(&d_y, N * sizeof(int));
	cudaMemcpy(d_y, y, N * sizeof(int), cudaMemcpyHostToDevice);
	
	curandState* devStates;
	cudaMalloc (&devStates, N * sizeof(curandState));
	
	// Run kernel on the GPU
	int blockSize = 32;
	int numBlocks = (N + blockSize - 1) / blockSize;
	
	setup_kernel<<<numBlocks, blockSize>>>(devStates, seed, N);
	
	cudaEventRecord(start);
	
	mcmc<<<numBlocks, blockSize>>>(num_timesteps, num_cells, num_genes, num_iterations, step_sizes, num_params, max_count, transcriptional_states, proposed_mrna_count, current_mrna_count, real_distributions, best_params, devStates);
	
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	printf("Elapsed seconds: %f\n", milliseconds/1000);
	
	FILE *outfile;
	string path_output_string = string(path_output_dir) + "mcmc_counts.csv";
	const char *path_output = path_output_string.c_str();
	printf("%s\n", path_output);
	outfile = fopen(path_output, "w");//create a file
	for (int i_gene = 0; i_gene < num_genes; i_gene++){
		fprintf(outfile, "%s,", gene_names[i_gene].c_str());
	    for (int i_cell = 0; i_cell < num_cells; i_cell++){
	        fprintf(outfile, "%i,", current_mrna_count[i_cell * num_genes + i_gene]);
	    }
	    fprintf(outfile,"\n");
	}
	fclose(outfile);
	
	// open params file
	FILE *outfile_parameters;
	string path_parameters_string = string(path_output_dir) + "mcmc_parameters.csv";
	const char *path_parameters_out = path_parameters_string.c_str();
	outfile_parameters = fopen(path_parameters_out, "w");//create a file
	printf("%s\n", path_parameters_out);
	fprintf(outfile_parameters, "gene,on,off,np,p_np,tx,\n");
	for (int i_gene = 0; i_gene < num_genes; i_gene++){
		fprintf(outfile, "%s,", gene_names[i_gene].c_str());
	    for (int i_param = 0; i_param < num_params; i_param++){
		    int i_param_gene = i_gene * num_params + i_param;
	        fprintf(outfile_parameters, "%i,", best_params[i_param_gene]);
	    }
	    fprintf(outfile_parameters,"\n");
	}
	fclose(outfile_parameters);

	// Free memory
	cudaFree(transcriptional_states);
	cudaFree(proposed_mrna_count);
	cudaFree(current_mrna_count);
	cudaFree(best_params);
	cudaFree(real_distributions);
	cudaFree(devStates);
	cudaFree(d_y);
	return 0;
}




