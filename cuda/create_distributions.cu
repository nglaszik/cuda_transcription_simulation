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
char mode[10]="mode";
int max_count = 400;
double step = 1.0;
double h = 4.0; // bandwidth for kde
double max_time = 3600.0; // 1 hour
int batch_size = 1000000;
double lower_limit = -5.0; // lower limit for parameters
double upper_limit = 2.0; // upper limit for parameters
double k_deg = -1.0;

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

		int res = gzwrite(file, data, size * sizeof(double));
		if (res == 0) {
			std::cerr << "Failed to write to file" << std::endl;
			return false;
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

tuple <fs::path, fs::path, fs::path> run_path_checks(fs::path path_outdir, int max_count, float max_time, float step, float h, float lower_limit, float upper_limit, float k_deg, fs::path mode_dir){
	// check to see if output_dir exists
	if (!dirExists(path_outdir.c_str())){
		printf("%s directory does not exist, please create\n", path_outdir.c_str());
		exit(0);
	}
	else {
		printf("%s directory exists\n", path_outdir.c_str());
	}
	
	string rundir_string = concatenate("max", max_count) + concatenate("_time", max_time) + concatenate("_step", step) + concatenate("_h", h) + concatenate("_lower", lower_limit) + concatenate("_upper", upper_limit) + concatenate("_deg", k_deg);
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
auto generate_kde_gpu(double *distributions, int *mrna_counts, int max_count, double h, int batch_size, int i_param_combination, int num_cells, int kde_granularity)
{
	const double x_0 = 0.0;
	const int Nx = max_count * kde_granularity;
	const double x_limit = (double)max_count;
	const double p = 1.0 / (h * max_count);
	const double hx = (x_limit - x_0)/(Nx - 1);
	
	for(int i_x = 0; i_x < Nx; ++i_x)
	{
		int i_dist = i_param_combination * Nx + i_x;
		double x = x_0 + i_x * hx;
		double sum = 0;
		for (int i_cell = 0; i_cell < num_cells; i_cell++) {
			int i_cell_param_combination = i_cell * batch_size + i_param_combination;
			//printf("filling distribution for cell %i, param combination %i, total index %i\n", i_cell, i_param_combination, i_cell_param_combination);
			//if (i_param_combination == 0) printf("%i,", mrna_counts[i_cell_param_combination]);
			sum += k_gpu((x - (double)mrna_counts[i_cell_param_combination]) / h);
		}
		distributions[i_dist] = p * sum;
	}
};

__global__
void generate_kde_gpu_parallel(double *distributions, int *mrna_counts, int max_count, double h, int num_genes, int num_cells, int kde_granularity)
{
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
		
	for (int i_gene = index; i_gene < num_genes; i_gene+=stride) {
		
		generate_kde_gpu(distributions, mrna_counts, max_count, h, num_genes, i_gene, num_cells, kde_granularity);
		
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
		distributions[i_dist] = p * sum;
	}
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
		else if (strcmp(argv[i], "-h") == 0){
			h=atof(argv[i+1]);
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
void simulate(double max_time, int num_cells, int i_batch, int batch_size, int num_combinations_current_batch, const int num_params, int max_count, double h, double *param_combinations, int *transcriptional_states, int *mrna_count, double *simulated_distributions, int kde_granularity, curandState* globalState){
	
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
			simulated_distributions[i_dist] = 0.0;
		}
		
		for (int i_cell = 0; i_cell < num_cells; i_cell++) {
			
			int i_cell_param_combination = i_cell * batch_size + i_param_combination;
			
			double r_np = generate(globalState, i_param_combination);
			// set an initial permanent off state of the gene in a particular cell
			if (r_np < param_combinations[i_param_combination * num_params + 4]) {
				transcriptional_states[i_cell_param_combination] = 2;																			// p_np
			}
			else {
				
				transcriptional_states[i_cell_param_combination] = 0;
				double time = 0.0;
				int iteration = 0;
				
				// no longer using iterations... need to make sure we get to steady state
				
				while (time < max_time && mrna_count[i_cell_param_combination] < max_count && (transcriptional_states[i_cell_param_combination] < 2 || mrna_count[i_cell_param_combination] > 0)) {
					
					double prob_np;
					double prob_switch;
					double prob_express;
					double prob_degrade;
					
					// degradation
					if (mrna_count[i_cell_param_combination] > 0){
						prob_degrade = (double)mrna_count[i_cell_param_combination] * param_combinations[i_param_combination * num_params + 5]; //degradation of mrna
					} else {
						prob_degrade = 0.0;
					}
					
					// transcription
					if (transcriptional_states[i_cell_param_combination] < 2){
						prob_np = param_combinations[i_param_combination * num_params + 3]; 	// k_np
						if (transcriptional_states[i_cell_param_combination] == 0){
							// gene is off
							prob_switch = param_combinations[i_param_combination * num_params + 0];	// k_on
							prob_express = 0.0;
						} else {
							// gene is on
							prob_switch = param_combinations[i_param_combination * num_params + 1];		// k_off
							prob_express = param_combinations[i_param_combination * num_params + 2];	// k_tx
						}
					} else {
						prob_np = 0.0;
						prob_switch = 0.0;
						prob_express = 0.0;
					}
					
					// determine which event occurs & timestep
					double dt = -log(generate(globalState, i_param_combination)) / (prob_np + prob_switch + prob_express + prob_degrade);
					double probs [4] = {prob_np, prob_switch, prob_express, prob_degrade};
					int len_probs = 4;
					double prob_event = generate(globalState, i_param_combination);
					int i_event = determine_event_alt(prob_event, probs, len_probs);
					
					//printf("dt_np: %f, dt_switch: %f, dt_express: %f, state: %i, event: %i\n", dt_np, dt_switch, dt_express, transcriptional_states[i_cell_param_combination], i_event);
					
					time = time + dt;
					iteration++;
					
					if (time < max_time){
						if (i_event == 0){
							// np
							transcriptional_states[i_cell_param_combination] = 2;
						}
						else if (i_event == 1){
							// switch
							if (transcriptional_states[i_cell_param_combination] == 1){
								transcriptional_states[i_cell_param_combination] = 0;
							} else {
								transcriptional_states[i_cell_param_combination] = 1;
							}
						} 
						else if (i_event == 2){
							// transcribe
							mrna_count[i_cell_param_combination]++;
						}
						else {
							// degrade
							mrna_count[i_cell_param_combination]--;
							//printf("Degradation occurred!\n");
						}
					}
				}
			}
		}
		
		generate_kde_gpu(simulated_distributions, mrna_count, max_count, h, batch_size, i_param_combination, num_cells, kde_granularity);
		
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

// nvcc /home/data/nlaszik/cuda_simulation/code/cuda/create_distributions.cu -o /home/data/nlaszik/cuda_simulation/code/cuda/build/create_distributions -lcurand -lboost_filesystem -lboost_system -lineinfo -lz

// /home/data/nlaszik/cuda_simulation/code/cuda/build/create_distributions -mt 10.0 -mc 400 -s 0.1 -h 2.0 -bs 1000000 -o /home/data/nlaszik/cuda_simulation/output/simulated -mode no_np -ll -3.0 -ul 3.0 -d 0.0

int main(int argc, char** argv)
{
	
	if(!parseCommand(argc, argv)) {
		cout<<"Error in arguments..\n";
		exit(0);
	}
	
	printf("max count: %i\n", max_count);
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
	fs::path path_kdes;
	fs::path path_parameters;
	fs::path path_counts;
	tie(path_kdes, path_parameters, path_counts) = run_path_checks(path_outdir, max_count, max_time, step, h, lower_limit, upper_limit, k_deg, path_mode);
	
	// test 0.0
	double test = pow(10.0, -DBL_MAX);
	if (test == 0.0){
		printf("0.0 test success: %f\n", test);
	} else {
		printf("0.0 test failed: %f\n", test);
		exit(0);
	}
	
	int kde_granularity = 2;
	
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
	
	int num_cells = 1000;
	
	printf("number of cells: %i\n", num_cells);
	
	int *transcriptional_states, *mrna_count;
	double *param_combinations, *simulated_distributions;
	
	// creating parameter combinations
	printf("creating parameter combinations...\n");
	const int num_params = 6;
	
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
	
	double param_lower_limits[num_params] = {lower_limit, 		lower_limit, 	lower_limit, 	lower_limit,	0.0, k_deg};
	double param_upper_limits[num_params] = {upper_limit, 		upper_limit, 	upper_limit, 	upper_limit,	1.0, k_deg};
	
	if (strcmp(mode, "no_np") == 0){
		param_lower_limits[3] = -DBL_MAX;
		param_upper_limits[3] = -DBL_MAX;
		param_lower_limits[4] = 0.0;
		param_upper_limits[4] = 0.0;
	}
	else if (strcmp(mode, "const") == 0){
		param_lower_limits[0] = DBL_MAX;
		param_upper_limits[0] = DBL_MAX;
		param_lower_limits[1] = -DBL_MAX;
		param_upper_limits[1] = -DBL_MAX;
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
		printf("Invalid mode provided. Please designate as 'const', 'no_np', 'no_knp', 'no_pnp', or 'full_model'\n");
		exit(0);
	}
	
	double step_sizes[num_params] = {(double)(step), (double)(step), (double)(step), (double)(step), (double)(step), (double)(step)};
	
	vector<vector<double>> param_matrix(num_params);
	// create parameters combinations
	int num_param_combinations = 1;
	for (int i_param = 0; i_param < num_params; i_param++){
		int param_size = 0;
		if (param_lower_limits[i_param] != -DBL_MAX && param_lower_limits[i_param] != DBL_MAX){
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
		else if (param_lower_limits[i_param] == DBL_MAX){
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
	
	vector<vector<double>> param_combinations_vector = cart_product(param_matrix);
	
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
	cudaMallocManaged(&simulated_distributions, batch_size * max_count * kde_granularity * sizeof(double));
	cudaMallocManaged(&param_combinations, batch_size * num_params * sizeof(double));
	
	printf("size of distributions: %i\n", batch_size * max_count * kde_granularity);
	
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
	
	// open counts file
	//FILE *outfile_kdes;
	//outfile_kdes = fopen(path_kdes.c_str(), "wb");
	GZipWriter writer(path_kdes.string());
	
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
					fprintf(outfile_parameters, "%f,", param_combinations[i_param_combination_param]);
				}
				fprintf(outfile_parameters, "\n");
				i_param_combination++;
			}
			else {
				break;
			}
		}
		
		printf("processing combination batch %i, num combinations: %i...\n", i_batch + 1, i_param_combination);
		
		simulate<<<numBlocks, blockSize>>>(max_time, num_cells, i_batch, batch_size, i_param_combination, num_params, max_count, h, param_combinations, transcriptional_states, mrna_count, simulated_distributions, kde_granularity, devStates);
		
		cudaEventRecord(stop);
		cudaDeviceSynchronize();
		
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		
		printf("Elapsed seconds: %f\n", milliseconds/1000);
		
		long unsigned int size_kde_batch = i_param_combination * max_count * kde_granularity;
		writer.write_batch(simulated_distributions, size_kde_batch);
		
		for (int i_batch_combination = 0; i_batch_combination < i_param_combination; i_batch_combination++){
			for (int i_cell = 0; i_cell < num_cells; i_cell++){
				int i_cell_param_combination = i_cell * batch_size + i_batch_combination;
				fprintf(outfile_counts, "%i,", mrna_count[i_cell_param_combination]);
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
