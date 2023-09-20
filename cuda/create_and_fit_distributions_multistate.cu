#include <iostream>
#include <math.h>
#include <chrono>
#include <random>
#include <curand.h>
#include <float.h>
#include <curand_kernel.h>
#include <string>
#include <array>
#include <sstream>
#include <map>
#include <fstream>
#include <iterator>

#include <sys/types.h>
#include <sys/stat.h>

#include <boost/filesystem.hpp>

using namespace std;
namespace fs = boost::filesystem;

// init some values
char path_output_dir[200]="path_output_dir";
char path_input_matrices[200]="path_input_matrices";
char mode[10]="mode";
int max_iterations = 5000;
int max_count = 400;
double step = 1.0;
double h = 4.0; // bandwidth for kde
double max_time = 3600.0; // 1 hour
int batch_size = 1000000;
double lower_limit = -5.0; // lower limit for parameters
double upper_limit = 2.0; // upper limit for parameters

string concatenate(std::string const& name, float i)
{
    stringstream s;
    s << name << i;
    return s.str();
}

int dirExists(const char *path)
{
    struct stat info;

    if(stat( path, &info ) != 0)
        return 0;
    else if(info.st_mode & S_IFDIR)
        return 1;
    else
        return 0;
}

tuple <fs::path, fs::path, fs::path> run_path_checks(fs::path path_outdir, float max_time, float step, float h, float lower_limit, float upper_limit, fs::path mode_dir){
	// check to see if output_dir exists
	if (!dirExists(path_outdir.c_str())){
		printf("%s directory does not exist, please create\n", path_outdir.c_str());
		exit(0);
	}
	else {
		printf("%s directory exists\n", path_outdir.c_str());
	}
	
	string rundir_string = concatenate("time", max_time) + concatenate("_step", step) + concatenate("_h", h) + concatenate("_lower", lower_limit) + concatenate("_upper", upper_limit);
	fs::path rundir (rundir_string);
	fs::path path_mode_dir = path_outdir / mode_dir;
	
	if (!dirExists(path_mode_dir.c_str())){
		printf("%s does not exist\n", path_mode_dir.c_str());
		int stat = mkdir(path_mode_dir.c_str(), 0775);
		if (!stat){
			printf("%s directory created successfully\n", path_mode_dir.c_str());
		}
	    else {
	        printf("%s directory could not be created\n", path_mode_dir.c_str());
	        exit(0);
		}
	} else {
		printf("%s directory exists\n", path_mode_dir.c_str());
	}
	
	fs::path path_run_dir = path_mode_dir / rundir;
	
	if (!dirExists(path_run_dir.c_str())){
		printf("%s does not exist\n", path_run_dir.c_str());
		int stat = mkdir(path_run_dir.c_str(), 0775);
		if (!stat){
			printf("%s directory created successfully\n", path_run_dir.c_str());
		}
	    else {
	        printf("%s directory could not be created\n", path_run_dir.c_str());
	        exit(0);
		}
	} else {
		printf("%s directory exists\n", path_run_dir.c_str());
	}
	
	fs::path filename_counts ("initial_counts.csv");
	fs::path filename_parameters ("initial_parameters.csv");
	fs::path filename_mses ("initial_mses.csv");
	
	fs::path path_counts = path_run_dir / filename_counts;
	fs::path path_parameters = path_run_dir / filename_parameters;
	fs::path path_mses = path_run_dir / filename_mses;
	
	FILE *outfile;
	printf("testing write to counts path...\n");
	outfile = fopen(path_counts.c_str(), "w");//create a file
	fprintf(outfile, "test");
	fclose(outfile);
	printf("write successful\n");
	
	return make_tuple(path_counts, path_parameters, path_mses);
	
}

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
			strcpy(path_input_matrices, argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-mode") == 0){
			strcpy(mode, argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-mi") == 0){
			max_iterations=atoi(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-mt") == 0){
			max_time=atof(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-s") == 0){
			step=atof(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-ll") == 0){
			lower_limit=atof(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-ul") == 0){
			upper_limit=atof(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-bs") == 0){
			batch_size=atoi(argv[i+1]);
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
double generate(curandState* globalState, int ind)
{
    curandState localState = globalState[ind];
    double RANDOM = curand_uniform_double( &localState );
    globalState[ind] = localState;
    return RANDOM;
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
    } else {
	    // transcribe
	    *i_event = 2;
	    return dt_express;
    }
}

__global__
void setup_kernel(curandState * state, unsigned long seed, int N)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < N) curand_init ( seed, id, 0, &state[id] );
}

__global__
void simulate(int max_iterations, double max_time, int num_cells, int num_genes, int i_batch, int batch_size, int num_combinations_current_batch, const int num_params, int max_count, double h, double *param_combinations, int *transcriptional_states, int *mrna_count, double *initial_distributions, double *real_distributions, double *best_mses, int *best_params, int *best_counts, curandState* globalState){
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
		
	for (int i_param_combination = index; i_param_combination < num_combinations_current_batch; i_param_combination+=stride) {
		
		if (i_param_combination % 100000 == 0){
			printf("processing batch combo %i...\n", i_param_combination);
		}
		
		// reset counts and states
		for (int i_cell = 0; i_cell < num_cells; i_cell++){
			int i_cell_param_combination = i_cell * batch_size + i_param_combination;
			mrna_count[i_cell_param_combination] = 0;
			transcriptional_states[i_cell_param_combination] = 0;
		}
		
		for (int i_count = 0; i_count < max_count; i_count++){
			int i_dist = i_param_combination * max_count + i_count;
			initial_distributions[i_dist] = 0.0;
		}
		
		for (int i_cell = 0; i_cell < num_cells; i_cell++) {
			
			int i_cell_param_combination = i_cell * batch_size + i_param_combination;
			
			transcriptional_states[i_cell_param_combination] = 0;
			double time = 0.0;
			int iteration = 0;
			
			while (iteration < max_iterations && time < max_time && mrna_count[i_cell_param_combination] < max_count && transcriptional_states[i_cell_param_combination] < 2) {
				
				double dt_switch;
				double dt_express;
				
				double dt_tx_change = -log(generate(globalState, i_param_combination))/param_combinations[i_param_combination * num_params + 3];	// k_tx change
				
				// create times to switch or transcribe
				if (transcriptional_states[i_cell_param_combination] == 0){
					// gene is off
					dt_switch = -log(generate(globalState, i_param_combination))/param_combinations[i_param_combination * num_params + 0];			// k_on
					dt_express = DBL_MAX;
				} else {
					// gene is on
					dt_switch = -log(generate(globalState, i_param_combination))/param_combinations[i_param_combination * num_params + 1];			// k_off
					dt_express = -log(generate(globalState, i_param_combination))/param_combinations[i_param_combination * num_params + 2];			// k_tx
				}
				
				// determine which event occurs first... basically whichever has the shortest time
				int i_event;
				double dt = determine_event(dt_tx_change, dt_switch, dt_express, &i_event);
				
				//printf("dt_np: %f, dt_switch: %f, dt_express: %f, state: %i, event: %i\n", dt_np, dt_switch, dt_express, transcriptional_states[i_cell_param_combination], i_event);
				
				time = time + dt;
				iteration++;
				
				// determine how much each parameter changes after this dt... this part is not stochastic
				param_combinations[i_param_combination * num_params + 2] *= (param_combinations[i_param_combination * num_params + 3] * dt)
				
				if (time < max_time){
					if (i_event == 0){
						// switch
						if (transcriptional_states[i_cell_param_combination] == 1){
							transcriptional_states[i_cell_param_combination] = 0;
						} else {
							transcriptional_states[i_cell_param_combination] = 1;
						}
					} else {
						// transcribe
						mrna_count[i_cell_param_combination]++;
					}
				}
			}
		}
		
		generate_kde_gpu(initial_distributions, mrna_count, max_count, h, batch_size, i_param_combination, num_cells);
		
		for (int i_gene = 0; i_gene < num_genes; i_gene++) {
		
			double mse = 0.0;
			int i_real_count;
			int i_initial_count;
			for (int i_count = 0; i_count < max_count; i_count++ ) {
				i_real_count = i_gene * max_count + i_count;
				i_initial_count = i_param_combination * max_count + i_count;
				mse += (initial_distributions[i_initial_count] - real_distributions[i_real_count]) * (initial_distributions[i_initial_count] - real_distributions[i_real_count]);
			}
			mse = mse / (double)max_count;
			if (mse < best_mses[i_gene]){
				best_mses[i_gene] = mse;
				best_params[i_gene] = i_batch * batch_size + i_param_combination;
				for (int i_cell = 0; i_cell < num_cells; i_cell++) {
					int i_cell_gene = i_cell * num_genes + i_gene;
					int i_cell_param_combination = i_cell * batch_size + i_param_combination;
					best_counts[i_cell_gene] = mrna_count[i_cell_param_combination];
				}
				//printf("new best mse found: %.16f for gene %i\n", mse, i_gene);
			}
		}
	}
}

vector<vector<double>> cart_product (const vector<vector<double>>& v) {
    vector<vector<double>> s = {{}};
    for (const auto& u : v) {
        vector<vector<double>> r;
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

// 3D to 1D and reverse
// x = i % width;
// y = (i / width)%height;
// z = i / (width*height);
//i = x + width*y + width*height*z;

// nvcc /home/data/nlaszik/cuda_simulation/code/cuda/create_and_fit_distributions_gillespie.cu -o /home/data/nlaszik/cuda_simulation/code/cuda/build/create_and_fit_distributions_gillespie -lcurand -lboost_filesystem -lboost_system -lineinfo

// SRP215251
// /home/data/nlaszik/cuda_simulation/code/cuda/build/create_and_fit_distributions_gillespie -mi 100000 -mt 50.0 -mc 400 -s 1.0 -h 4.0 -bs 2000000 -i /home/data/Shared/shared_datasets/sc-rna-seq/SRP215251/figures/wt/tpm_filtered.csv -o /home/data/nlaszik/cuda_simulation/output/SRP215251/WT -mode no_np -ll -5.0 -ul 1.0

// /home/data/nlaszik/cuda_simulation/code/cuda/build/create_and_fit_distributions_gillespie -mi 100000 -mt 50.0 -mc 400 -s 1.0 -h 4.0 -bs 2000000 -i /home/data/Shared/shared_datasets/sc-rna-seq/SRP215251/figures/dko/tpm_filtered.csv -o /home/data/nlaszik/cuda_simulation/output/SRP215251/DKO -mode no_np -ll -5.0 -ul 1.0

// SRP313343
// /home/data/nlaszik/cuda_simulation/code/cuda/build/create_and_fit_distributions_gillespie -mi 100000 -mt 50.0 -mc 400 -s 1.0 -h 4.0 -bs 2000000 -i /home/data/Shared/shared_datasets/sc-rna-seq/SRP313343/seurat/transcript_counts/srr14139729_transcript_counts.filtered.norm.csv -o /home/data/nlaszik/cuda_simulation/output/SRP313343/SRR14139729 -mode no_np -ll -5.0 -ul 1.0

// SRP299892
// smaller batch size due to higher # of cells?
// /home/data/nlaszik/cuda_simulation/code/cuda/build/create_and_fit_distributions_gillespie -mi 100000 -mt 50.0 -mc 400 -s 0.02 -h 4.0 -bs 1000000 -i /home/data/Shared/shared_datasets/sc-rna-seq/SRP299892/seurat/transcript_counts/srr13336770_transcript_counts.filtered.norm.csv -o /home/data/nlaszik/cuda_simulation/output/SRP299892 -mode no_np -ll -5.0 -ul 1.0

// DKO
// /home/data/nlaszik/cuda_simulation/code/cuda/build/create_and_fit_distributions_gillespie -mi 100000 -mt 50.0 -mc 150 -s 1.0 -h 4.0 -bs 1000000 -i /home/data/Shared/shared_datasets/sc-rna-seq/dko_hesc/seurat/transcript_counts/rep2_transcript_counts.filtered.norm.csv -o /home/data/nlaszik/cuda_simulation/output/dko_hesc/rep2 -mode no_np -ll -5.0 -ul 1.0

// DKO same median
// /home/data/nlaszik/cuda_simulation/code/cuda/build/create_and_fit_distributions_gillespie -mi 100000 -mt 50.0 -mc 150 -s 1.0 -h 4.0 -bs 1000000 -i /home/data/Shared/shared_datasets/sc-rna-seq/dko_hesc/seurat/set_median/rep1_transcript_counts.filtered.norm.csv -o /home/data/nlaszik/cuda_simulation/output/dko_hesc/rep1_gillespie_set_median/ -mode no_np -ll -5.0 -ul 1.0

// Andrew's data
// /home/data/nlaszik/cuda_simulation/code/cuda/build/create_and_fit_distributions_gillespie -mi 100000 -mt 50.0 -mc 400 -s 1.0 -h 4.0 -bs 1000000 -i /home/data/Shared/AQPhan/norm_counts.csv -o /home/data/nlaszik/cuda_simulation/output/aqphan/ -mode no_np -ll -5.0 -ul 1.0

// SRP364225
// /home/data/nlaszik/cuda_simulation/code/cuda/build/create_and_fit_distributions_gillespie -mi 100000 -mt 50.0 -mc 400 -s 1.0 -h 4.0 -bs 1000000 -i /home/data/Shared/shared_datasets/sc-rna-seq/SRP364225/seurat/SRR18335026_transcript_counts.filtered.norm.csv -o /home/data/nlaszik/cuda_simulation/output/SRP364225/SRR18335026_norm/ -mode no_np -ll -5.0 -ul 1.0


int main(int argc, char** argv)
{
	
	if(!parseCommand(argc, argv)) {
	    cout<<"Error in arguments..\n";
	    exit(0);
	}
	
	printf("max count: %i\n", max_count);
	printf("max iterations: %i\n", max_iterations);
	printf("max time in seconds: %f\n", max_time);
	printf("batch size: %i\n", batch_size);
	printf("step size: %f\n", step);
	printf("h: %f\n", h);
	
	if (strcmp(mode, "mode") == 0){
		printf("Please provide a mode. Options: no_np, no_knp, no_pnp, full_model.\n");
		exit(0);
	}
	
	// check directories
	fs::path path_outdir (path_output_dir);
	fs::path path_mode (mode);
	fs::path path_counts;
	fs::path path_parameters;
	fs::path path_mses;
	tie(path_counts, path_parameters, path_mses) = run_path_checks(path_outdir, max_time, step, h, lower_limit, upper_limit, path_mode);
	
	// test 0.0
	double test = pow(10.0, -DBL_MAX);
	if (test == 0.0){
		printf("0.0 test success: %f\n", test);
	} else {
		printf("0.0 test failed: %f\n", test);
		exit(0);
	}
	
	// test inf
	double inf_test = 10.0 / 0.0;
	printf("inf test: %f\n", inf_test);
	
	// check gpu memory
	float free_m,total_m,used_m;
	size_t free_t,total_t;
	cudaMemGetInfo(&free_t,&total_t);
	free_m =(float)free_t/1048576.0;
	total_m=(float)total_t/1048576.0;
	used_m=total_m-free_m;
	printf ("mem free %f MB, mem total %f MB, mem used %f MB\n", free_m, total_m, used_m);
	
	// load real distributions from cell count matrix
    // also make initial guesses for parameters
	ifstream infile(path_input_matrices);
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
	
	int *best_counts, *best_params;
	int *real_mrna_count = new int[num_cells * num_genes];
	double *real_distributions, *initial_distributions, *best_mses;
	
	cudaMallocManaged(&best_mses, num_genes * sizeof(double));
	cudaMallocManaged(&best_params, num_genes * sizeof(int));
	cudaMallocManaged(&real_distributions, num_genes * max_count * sizeof(double));
	cudaMallocManaged(&best_counts, num_genes * num_cells * sizeof(int));
	printf("able to initialize memory on gpu...\n");
	
	// init best mses and params... not too necessary for params but w/e
	for (int i_fill = 0; i_fill < num_genes; i_fill++) {
		best_mses[i_fill] = 1.0;
		best_params[i_fill] = 0;
	}
	printf("able to assign memory on gpu...\n");
	
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

	
	int *transcriptional_states, *mrna_count;
	double *param_combinations;
	
	// creating parameter combinations
	printf("creating parameter combinations...\n");
	const int num_params = 5;
	
	// these are rates / second 
	// max rate should be once every 5 seconds = 720.0/hour = 0.2/sec... for high range, maybe instead just do linear rate changes 0.195, 0.19, 0.185, ... etc
	// for low range, next is maybe 0.19, 0.18, 0.1
	
	// min rate should 5.0/hour = 0.005/sec... we can actually do smaller increments it seems maybe .0025?
	
	// k_on, k_off, k_tx, k_np, p_np
	
	// perhaps we can do this on log scale?
	
	// scale between k_np and k_on/k_off should be quite different? 
	// also play around with making bimodal distributions with small k_on and large k_off
	// transformers for gene/gene interactions? a la protein-protein interaction?
	
	// since log range, we start with negatives
	
	double param_lower_limits[num_params] = {lower_limit, 		lower_limit, 	lower_limit, 	lower_limit,	0.0};
	double param_upper_limits[num_params] = {upper_limit, 		upper_limit, 	upper_limit, 	upper_limit,	1.0};
	
	if (strcmp(mode, "no_np") == 0){
		param_lower_limits[3] = -DBL_MAX;
		param_upper_limits[3] = -DBL_MAX;
		param_lower_limits[4] = 0.0;
		param_upper_limits[4] = 0.0;
	}
	else if (strcmp(mode, "no_knp") == 0){
		param_lower_limits[3] = -DBL_MAX;
		param_upper_limits[3] = -DBL_MAX;
	}
	else if (strcmp(mode, "no_pnp") == 0){
		param_lower_limits[4] = 0.0;
		param_upper_limits[4] = 0.0;
	}
	else if (strcmp(mode, "full_model") == 0){
		printf("All parameters selected.\n");
	}
	else {
		printf("Invalid mode provided. Please designate as 'no_np', 'no_knp', 'no_pnp', or 'full_model'\n");
		exit(0);
	}
	
	double step_sizes[num_params] = {(double)(step), (double)(step), (double)(step), (double)(step), (double)(step)};
	
	vector<vector<double>> param_matrix(num_params);
	// create parameters combinations
	int num_param_combintations = 1;
	for (int i_param = 0; i_param < num_params; i_param++){
		int param_size = 0;
		if (param_lower_limits[i_param] != -DBL_MAX){
			for (double param = param_lower_limits[i_param]; param <= param_upper_limits[i_param]; param += step_sizes[i_param]){
				if (i_param == 4){
					// p_np is a probability, not a rate
					param_matrix[i_param].push_back(param);
				}
				else {
					// rate, so use log
					param_matrix[i_param].push_back(pow(10.0, param));
				}
				param_size++;
			}
		}
		else {
			param_matrix[i_param].push_back(0.0);
			param_size++;
		}
		num_param_combintations *= param_size;
	}
	
	printf("number of param combinations: %i\n", num_param_combintations);
	
	vector<vector<double>> param_combinations_vector = cart_product(param_matrix);
	
	int num_batches;
	if (num_param_combintations <= batch_size) {
		batch_size = num_param_combintations;
		num_batches = 1;
	}
	else {
		num_batches = (int)ceil(num_param_combintations / batch_size) + 1;
	}
	
	printf("num batches: %i, final batch size: %i\n", num_batches, batch_size);
	
	cudaMallocManaged(&transcriptional_states, batch_size * num_cells * sizeof(int));
	cudaMallocManaged(&mrna_count, batch_size * num_cells * sizeof(int));
	cudaMallocManaged(&initial_distributions, batch_size * max_count * sizeof(double));
	cudaMallocManaged(&param_combinations, batch_size * num_params * sizeof(double));
	
	// set up to fit on gpu
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
	default_random_engine generator (seed);
	
	// setup and allocate memory for curand
	int N = batch_size;
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
	
	for (int i_batch = 0; i_batch < num_batches; i_batch++){
		
		// assign params vector to gpu memory
		int i_param_combination = 0;
		for (int i_batch_combination = i_batch * batch_size; i_batch_combination < (i_batch + 1) * batch_size; i_batch_combination++){
			if (i_batch_combination < num_param_combintations) {
				for (int i_param = 0; i_param < num_params; i_param++){
					int i_param_combination_param = i_param_combination * num_params + i_param;
					param_combinations[i_param_combination_param] = param_combinations_vector[i_batch_combination][i_param];
				}
				i_param_combination++;
			}
			else {
				break;
			}
		}
		
		printf("processing combination batch %i, num combinations: %i...\n", i_batch + 1, i_param_combination);
		
		simulate<<<numBlocks, blockSize>>>(max_iterations, max_time, num_cells, num_genes, i_batch, batch_size, i_param_combination, num_params, max_count, h, param_combinations, transcriptional_states, mrna_count, initial_distributions, real_distributions, best_mses, best_params, best_counts, devStates);
		
		cudaEventRecord(stop);
		cudaDeviceSynchronize();
		
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		
		printf("Elapsed seconds: %f\n", milliseconds/1000);
		
	}
	
	FILE *outfile_counts;
	outfile_counts = fopen(path_counts.c_str(), "w");//create a file
	printf("writing %s\n", path_counts.c_str());
	for (int i_gene = 0; i_gene < num_genes; i_gene++){
		fprintf(outfile_counts, "%s,", gene_names[i_gene].c_str());
	    for (int i_cell = 0; i_cell < num_cells; i_cell++){
	        fprintf(outfile_counts, "%i,", best_counts[i_cell * num_genes + i_gene]);
	    }
	    fprintf(outfile_counts,"\n");
	}
	fclose(outfile_counts);
	
	// open params file
	FILE *outfile_parameters;
	outfile_parameters = fopen(path_parameters.c_str(), "w");//create a file
	printf("writing %s\n", path_parameters.c_str());
	fprintf(outfile_parameters, "gene,on,off,tx,np,p_np,\n");
	for (int i_gene = 0; i_gene < num_genes; i_gene++){
		fprintf(outfile_parameters, "%s,", gene_names[i_gene].c_str());
	    for (int i_param = 0; i_param < num_params; i_param++){
	        fprintf(outfile_parameters, "%.16f,", param_combinations_vector[best_params[i_gene]][i_param]);
	    }
	    fprintf(outfile_parameters,"\n");
	}
	fclose(outfile_parameters);
	
	// open params file
	FILE *outfile_mse;
	outfile_mse = fopen(path_mses.c_str(), "w");//create a file
	printf("writing %s\n", path_mses.c_str());
	for (int i_gene = 0; i_gene < num_genes; i_gene++){
		fprintf(outfile_mse, "%s,%.16f\n", gene_names[i_gene].c_str(), best_mses[i_gene]);
	}
	fclose(outfile_mse);

	// Free memory
	param_combinations_vector.clear();
	cudaFree(param_combinations);
	cudaFree(initial_distributions);
	cudaFree(transcriptional_states);
	cudaFree(mrna_count);
	cudaFree(best_mses);
	cudaFree(best_params);
	cudaFree(real_distributions);
	cudaFree(best_counts);
	cudaFree(devStates);
	cudaFree(d_y);
	return 0;
}




