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
#include <zlib.h>

#include <boost/filesystem.hpp>

using namespace std;
namespace fs = boost::filesystem;

// init some values
char path_output_dir[200]="path_output_dir";
char path_params[200]="path_params";
char mode[10]="mode";
int max_count = 400;
double step = 1.0;
double lower_limit = -5.0; // lower limit for parameters
double upper_limit = 2.0; // upper limit for parameters
double max_time = 10.0; // 10 hours
int batch_size = 1000000;
double k_deg = 0.0;
int num_cells = 1000;
int num_decimals = 0;

class GZipWriter {
public:
	GZipWriter(const std::string& filename) {
		file = gzopen(filename.c_str(), "wb");
		if (!file) {
			std::cerr << "Failed to open file: " << filename << std::endl;
		}
	}

	~GZipWriter() {
		if (file) {
			gzclose(file);
		}
	}

	bool write_batch(const double* data, size_t size) {
		if (!file) {
			return false;
		}
		
		const size_t chunk_size = (128 * 1024 * 1024) / sizeof(double); // 128MB
		size_t chunks = size / chunk_size;
		size_t remainder = size % chunk_size;
		
		for (size_t i = 0; i < chunks; ++i) {
			int res = gzwrite(file, data + i * chunk_size, chunk_size * sizeof(double));
			if (res == 0) {
				std::cerr << "Failed to write to file" << std::endl;
				return false;
			}
		}

		if (remainder > 0) {
			int res = gzwrite(file, data + chunks * chunk_size, remainder * sizeof(double));
			if (res == 0) {
				std::cerr << "Failed to write remainder to file" << std::endl;
				return false;
			}
		}

		return true;
	}

private:
	gzFile file = nullptr;
};

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

tuple <fs::path, fs::path, fs::path> run_path_checks(fs::path path_outdir, int max_count, float max_time, float step, float lower_limit, float upper_limit, float k_deg, int num_cells, fs::path mode_dir, bool create_new_params){
	// check to see if output_dir exists
	if (!dirExists(path_outdir.c_str())){
		printf("%s directory does not exist, please create\n", path_outdir.c_str());
		exit(0);
	}
	else {
		printf("%s directory exists\n", path_outdir.c_str());
	}
	
	string rundir_string;
	if (create_new_params){
		rundir_string = concatenate("ncell", num_cells) + concatenate("_max", max_count) + concatenate("_time", max_time) + concatenate("_step", step) + concatenate("_lower", lower_limit) + concatenate("_upper", upper_limit) + concatenate("_deg", k_deg);
	} else {
		rundir_string = concatenate("ncell", num_cells) + concatenate("_max", max_count) + concatenate("_time", max_time) + "_custom_parameters";
	}
	
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
	
	fs::path filename_kdes ("kdes.bin.gz");
	fs::path filename_parameters ("parameters.csv");
	fs::path filename_counts ("counts.csv");
	
	fs::path path_kdes = path_run_dir / filename_kdes;
	fs::path path_parameters = path_run_dir / filename_parameters;
	fs::path path_counts = path_run_dir / filename_counts;
	
	return make_tuple(path_kdes, path_parameters, path_counts);
	
}

__device__
auto generate_hist_gpu(double *distributions, int *mrna_counts, int max_count, int batch_size, int i_param_combination, int num_cells)
{
	for (int i_cell = 0; i_cell < num_cells; i_cell++) {
		int i_cell_param_combination = i_cell * batch_size + i_param_combination;
		int count = mrna_counts[i_cell_param_combination];
		if (count <= max_count){
			int i_dist = i_param_combination * max_count + count;
			distributions[i_dist] += 1.0 / (double)num_cells;
		}
	}
};

__global__
void generate_hist_gpu_parallel(double *distributions, int *mrna_counts, int max_count, int num_genes, int num_cells)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
		
	for (int i_gene = index; i_gene < num_genes; i_gene+=stride) {
		
		generate_hist_gpu(distributions, mrna_counts, max_count, num_genes, i_gene, num_cells);
		
	}
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
		if (strcmp(argv[i], "-p") == 0){
			strcpy(path_params, argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-mode") == 0){
			strcpy(mode, argv[i+1]);
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
		else if (strcmp(argv[i], "-d") == 0){
			k_deg=atof(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-mc") == 0){
			max_count=atoi(argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-ncell") == 0){
			num_cells=atoi(argv[i+1]);
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
int determine_event_alt(double prob_event, double *probs, int len_probs)
{
	double sum_probs = 0.0;
	for(int i=0; i < len_probs; i++){
		sum_probs += probs[i];
	}
	for(int i=0; i < len_probs; i++){
		probs[i] = probs[i] / sum_probs;
	}
	
	double rand_sum = 0.0;
	int i = 0;
	while (rand_sum < prob_event){
		rand_sum += probs[i];
		i += 1;
	}
	return i - 1;
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
void simulate(double max_time, int num_cells, int i_batch, int batch_size, int num_combinations_current_batch, const int num_params, int max_count, double *param_combinations, int *transcriptional_states, int *mrna_count, double *simulated_distributions, curandState* globalState){
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
		
	for (int i_param_combination = index; i_param_combination < num_combinations_current_batch; i_param_combination+=stride) {
		
		if (i_param_combination % 10000 == 0){
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
			simulated_distributions[i_dist] = 0.0;
		}
		
		for (int i_cell = 0; i_cell < num_cells; i_cell++) {
			
			int i_cell_param_combination = i_cell * batch_size + i_param_combination;
				
			transcriptional_states[i_cell_param_combination] = 0;
			double time = 0.0;
			int iteration = 0;
			
			// no longer using iterations... need to make sure we get to steady state
			
			while (time < max_time && mrna_count[i_cell_param_combination] < max_count - 1) {
				
				double prob_switch;
				double prob_express;
				double prob_degrade;
				
				// degradation
				if (mrna_count[i_cell_param_combination] > 0){
					prob_degrade = (double)mrna_count[i_cell_param_combination] * param_combinations[i_param_combination * num_params + 3]; //degradation of mrna
				} else {
					prob_degrade = 0.0;
				}
				
				// here we would have to check any interacting mrna products (for now, we can include protein later) and modify some parameter (likely k_on)
				// we actually can't do this since cells aren't running in parallel... also we are not parallelizing over genes but parameter combinations
				// we would need to change architecture of code completely. a simulated dataset would need to have everything defined beforehand
				// actually would make for a lot fewer combinations
				
				// this might actually be kind of a logic puzzle... you can only get the correct distribution through
				
				// workflow:
				// 1. Estimate params for some dataset (since gene dist is typically poisson/lognormal we can make some assumptions)
				// 2. Get interactions through spearman correlations and use these to inform baseline interactions (maybe take from TKO dataset). This is good since maybe we can also use silencing interactions
				// 3. Get the highest interactions here... all others can be ignored for efficiency
				// 4. We need to consider directionality... this could be tricky... essentially the number of gene pair interactions squared? Very High N
				// 3. We need to parallelize over cells. Each full run of a set of cells is one parameter combination essentially
				// 4. still will take a long time since we need to loop over each other gene... using this we create a combined total modification of k_on (maybe just add them up?)
				// 5. We can alter the interactions in some way once a run is completed and then 
				
				// transcription
				if (transcriptional_states[i_cell_param_combination] == 0){
					// gene is off
					prob_switch = param_combinations[i_param_combination * num_params + 0];	// k_on
					prob_express = 0.0;
				} else {
					// gene is on
					prob_switch = param_combinations[i_param_combination * num_params + 1];		// k_off
					prob_express = param_combinations[i_param_combination * num_params + 2];	// k_tx
				}
				
				// determine which event occurs & timestep
				double dt = -log(generate(globalState, i_param_combination)) / (prob_np + prob_switch + prob_express + prob_degrade);
				double probs [3] = {prob_switch, prob_express, prob_degrade};
				int len_probs = 3;
				double prob_event = generate(globalState, i_param_combination);
				int i_event = determine_event_alt(prob_event, probs, len_probs);
				
				//printf("dt_np: %f, dt_switch: %f, dt_express: %f, state: %i, event: %i\n", dt_np, dt_switch, dt_express, transcriptional_states[i_cell_param_combination], i_event);
				
				time = time + dt;
				iteration++;
				
				if (time < max_time){
					if (i_event == 0){
						// switch
						if (transcriptional_states[i_cell_param_combination] == 1){
							transcriptional_states[i_cell_param_combination] = 0;
						} else {
							transcriptional_states[i_cell_param_combination] = 1;
						}
					} 
					else if (i_event == 1){
						// transcribe
						mrna_count[i_cell_param_combination]++;
					}
					else {
						// degrade
						mrna_count[i_cell_param_combination]--;
					}
				}
			}
		}
		
		generate_hist_gpu(simulated_distributions, mrna_count, max_count, batch_size, i_param_combination, num_cells);
		
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

// nvcc /home/data/nlaszik/cuda_simulation/code/cuda/create_distributions_hist.cu -o /home/data/nlaszik/cuda_simulation/code/cuda/build/create_distributions_hist -lcurand -lboost_filesystem -lboost_system -lineinfo -lz

// /home/data/nlaszik/cuda_simulation/code/cuda/build/create_distributions_hist -mt 10.0 -mc 400 -s 0.1 -bs 1000000 -o /home/data/nlaszik/cuda_simulation/output/simulated_hist -mode no_np -ll -3.0 -ul 3.0 -d 0.0 -ncell 1000

// /home/data/nlaszik/cuda_simulation/code/cuda/build/create_distributions_hist -mt 640.0 -mc 400 -bs 1000000 -o /home/data/nlaszik/cuda_simulation/output/simulated_hist -mode no_np -ncell 3000 -p /home/data/nlaszik/cuda_simulation/output/SRP299892_junhao_cme/parameters.csv

// /home/data/nlaszik/cuda_simulation/code/cuda/build/create_distributions_hist -mt 640.0 -mc 400 -bs 1000000 -o /home/data/nlaszik/cuda_simulation/output/simulated_hist_test -mode no_np -ncell 1000 -p /home/data/nlaszik/cuda_simulation/output/SRP299892_junhao_cme/test_parameters.csv

int main(int argc, char** argv)
{
	
	if(!parseCommand(argc, argv)) {
		cout<<"Error in arguments..\n";
		exit(0);
	}
	
	printf("max count: %i\n", max_count);
	printf("max time in seconds: %f\n", max_time);
	printf("batch size: %i\n", batch_size);
	printf("number of cells: %i\n", num_cells);
	printf("step size: %f\n", step);
	
	if (strcmp(mode, "mode") == 0){
		printf("Please provide a mode. Options: no_np, no_knp, no_pnp, full_model.\n");
		exit(0);
	}
	
	if (step < 0.0000001){
		printf("Step size too small. Please choose a value larger than 0.0000001\n");
		exit(0);
	}
	
	// check directories
	fs::path path_outdir (path_output_dir);
	fs::path path_mode (mode);
	fs::path path_kdes;
	fs::path path_parameters;
	fs::path path_counts;
	
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
	
	int *transcriptional_states, *mrna_count;
	double *param_combinations, *simulated_distributions;
	
	// creating parameter combinations
	const int num_params = 6;
	vector<vector<double>> param_combinations_vector;
	int num_param_combinations;
	
	// these are rates / second 
	// max rate should be once every 5 seconds = 720.0/hour = 0.2/sec... for high range, maybe instead just do linear rate changes 0.195, 0.19, 0.185, ... etc
	// for low range, next is maybe 0.19, 0.18, 0.1
	
	// min rate should 5.0/hour = 0.005/sec... we can actually do smaller increments it seems maybe .0025?
	
	// k_on, k_off, k_tx, k_np, p_np, k_deg
	
	// perhaps we can do this on log scale?
	
	// scale between k_np and k_on/k_off should be quite different? 
	// also play around with making bimodal distributions with small k_on and large k_off
	// transformers for gene/gene interactions? a la protein-protein interaction?
	
	// since log range, we start with negatives
	if (strcmp(path_params, "path_params") == 0){
		
		tie(path_kdes, path_parameters, path_counts) = run_path_checks(path_outdir, max_count, max_time, step, lower_limit, upper_limit, k_deg, num_cells, path_mode, true);
		
		printf("creating parameter combinations...\n");
		
		double multiplier = 10000000;
		
		int int_lower_limit = (int)(lower_limit * multiplier);
		int int_upper_limit = (int)(upper_limit * multiplier);
		
		int param_lower_limits[num_params] = {int_lower_limit, 		int_lower_limit, 	int_lower_limit, 	int_lower_limit,	(int)(0.0 * multiplier), (int)(k_deg * multiplier)};
		int param_upper_limits[num_params] = {int_upper_limit, 		int_upper_limit, 	int_upper_limit, 	int_upper_limit,	(int)(1.0 * multiplier), (int)(k_deg * multiplier)};
		
		if (strcmp(mode, "no_np") == 0){
			param_lower_limits[3] = -INT_MAX;
			param_upper_limits[3] = -INT_MAX;
			param_lower_limits[4] = 0;
			param_upper_limits[4] = 0;
		}
		else if (strcmp(mode, "const") == 0){
			param_lower_limits[0] = INT_MAX;
			param_upper_limits[0] = INT_MAX;
			param_lower_limits[1] = -INT_MAX;
			param_upper_limits[1] = -INT_MAX;
			param_lower_limits[3] = -INT_MAX;
			param_upper_limits[3] = -INT_MAX;
			param_lower_limits[4] = 0;
			param_upper_limits[4] = 0;
		}
		else if (strcmp(mode, "no_knp") == 0){
			param_lower_limits[3] = -INT_MAX;
			param_upper_limits[3] = -INT_MAX;
		}
		else if (strcmp(mode, "no_pnp") == 0){
			param_lower_limits[4] = 0;
			param_upper_limits[4] = 0;
		}
		else if (strcmp(mode, "full_model") == 0){
			printf("All parameters selected.\n");
		}
		else {
			printf("Invalid mode provided. Please designate as 'const', 'no_np', 'no_knp', 'no_pnp', or 'full_model'\n");
			exit(0);
		}
		
		int int_step = (int)(step * multiplier);
		int step_sizes[num_params] = {int_step, int_step, int_step, int_step, int_step, int_step};
		
		vector<vector<double>> param_matrix(num_params);
		// create parameters combinations
		num_param_combinations = 1;
		for (int i_param = 0; i_param < num_params; i_param++){
			int param_size = 0;
			if (param_lower_limits[i_param] != -INT_MAX && param_lower_limits[i_param] != INT_MAX){
				for (int param = param_lower_limits[i_param]; param <= param_upper_limits[i_param]; param += step_sizes[i_param]){
					double param_dbl = (double)param / multiplier;
					if (i_param == 4){
						// p_np is a probability, not a rate
						param_matrix[i_param].push_back(param_dbl);
					}
					else {
						// rate, so use log
						param_matrix[i_param].push_back(pow(10.0, param_dbl));
					}
					param_size++;
				}
			}
			else if (param_lower_limits[i_param] == INT_MAX){
				param_matrix[i_param].push_back(DBL_MAX);
				param_size++;
			}
			else {
				param_matrix[i_param].push_back(0.0);
				param_size++;
			}
			num_param_combinations *= param_size;
		}
		
		printf("number of param combinations: %i\n", num_param_combinations);
		
		param_combinations_vector = cart_product(param_matrix);
	
	} else {
		
		tie(path_kdes, path_parameters, path_counts) = run_path_checks(path_outdir, max_count, max_time, step, lower_limit, upper_limit, k_deg, num_cells, path_mode, false);
		
		printf("loading existing parameter combinations...\n");
		ifstream paramsfile(path_params);
		vector<string> rows_params;
		std::string line_params;
		std::size_t last_pos = 0;
		std::size_t pos = 0;
		
		while (getline(paramsfile, line_params, '\n'))
		{
			rows_params.push_back(line_params); //Get each line of the file as a string
		}
		num_param_combinations = rows_params.size() - 1;
		printf("number of param combinations: %i\n", num_param_combinations);
		
		for (int i=1; i<rows_params.size(); ++i){
			// the first thing will be a string cell_id
			last_pos = (size_t)-1;
			pos = 0;
			
			vector<double> holder_vector;
			param_combinations_vector.push_back(holder_vector);
			
			int i_param = 0;
			double parameter = 0.0;
			while (pos != std::string::npos){
				pos = rows_params[i].find(",", last_pos + 1);
				if (rows_params[i].substr(last_pos + 1, pos - last_pos - 1).empty()) {
					parameter = 0.0;
				}
				else {
					parameter = stof(rows_params[i].substr(last_pos + 1, pos - last_pos - 1));
					param_combinations_vector.back().push_back(parameter);
					i_param++;
				}
				// initialize param values
				last_pos = pos;
			}
		}
	}
	
	int num_batches;
	if (num_param_combinations <= batch_size) {
		batch_size = num_param_combinations;
		num_batches = 1;
	}
	else {
		num_batches = (int)ceil(num_param_combinations / batch_size) + 1;
	}
	
	printf("num batches: %i, final batch size: %i\n", num_batches, batch_size);
	
	cudaMallocManaged(&transcriptional_states, batch_size * num_cells * sizeof(int));
	cudaMallocManaged(&mrna_count, batch_size * num_cells * sizeof(int));
	cudaMallocManaged(&simulated_distributions, batch_size * max_count * sizeof(double));
	cudaMallocManaged(&param_combinations, batch_size * num_params * sizeof(double));
	
	printf("size of distributions: %i\n", batch_size * max_count);
	
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
	
	// open kde file
	GZipWriter writer(path_kdes.string());
	
	// open counts file
	FILE *outfile_counts;
	outfile_counts = fopen(path_counts.c_str(), "w");
	
	// open params file
	FILE *outfile_parameters;
	outfile_parameters = fopen(path_parameters.c_str(), "w");//create a file
	printf("writing %s\n", path_parameters.c_str());
	fprintf(outfile_parameters, "on,off,tx,np,p_np,deg,\n");
	
	for (int i_batch = 0; i_batch < num_batches; i_batch++){
		
		// assign params vector to gpu memory
		int i_param_combination = 0;
		for (int i_batch_combination = i_batch * batch_size; i_batch_combination < (i_batch + 1) * batch_size; i_batch_combination++){
			if (i_batch_combination < num_param_combinations) {
				for (int i_param = 0; i_param < num_params; i_param++){
					int i_param_combination_param = i_param_combination * num_params + i_param;
					param_combinations[i_param_combination_param] = param_combinations_vector[i_batch_combination][i_param];
					fprintf(outfile_parameters, "%.16f,", param_combinations[i_param_combination_param]);
				}
				fprintf(outfile_parameters, "\n");
				i_param_combination++;
			}
			else {
				break;
			}
		}
		
		printf("processing combination batch %i, num combinations: %i...\n", i_batch + 1, i_param_combination);
		
		simulate<<<numBlocks, blockSize>>>(max_time, num_cells, i_batch, batch_size, i_param_combination, num_params, max_count, param_combinations, transcriptional_states, mrna_count, simulated_distributions, devStates);
		
		cudaEventRecord(stop);
		cudaDeviceSynchronize();
		
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		
		printf("Elapsed seconds: %f\n", milliseconds/1000);
		
		long unsigned int size_kde_batch = i_param_combination * max_count;
		writer.write_batch(simulated_distributions, size_kde_batch);
		
		for (int i_batch_combination = 0; i_batch_combination < i_param_combination; i_batch_combination++){
			map<int, int> counts;
			for (int i_cell = 0; i_cell < num_cells; i_cell++){
				int i_cell_param_combination = i_cell * batch_size + i_batch_combination;
				int count = mrna_count[i_cell_param_combination];
				if (counts.find(count) == counts.end()) {
					counts[mrna_count[i_cell_param_combination]] = 1;
				} else {
					counts[mrna_count[i_cell_param_combination]]++;
				}
			}
			for (auto i : counts){
				fprintf(outfile_counts, "%i:%i,", i.first, i.second);
			}
			fprintf(outfile_counts, "\n");
		}
	}

	// Free memory
	param_combinations_vector.clear();
	cudaFree(param_combinations);
	cudaFree(simulated_distributions);
	cudaFree(transcriptional_states);
	cudaFree(mrna_count);
	cudaFree(devStates);
	cudaFree(d_y);
	return 0;
}
