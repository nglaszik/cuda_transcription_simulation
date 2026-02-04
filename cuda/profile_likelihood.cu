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
#include <algorithm>
#include <regex>

#include <sys/types.h>
#include <sys/stat.h>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>

using namespace std;
namespace fs = boost::filesystem;

// init some values
char path_output_dir[200]="path_output_dir";
char path_input_counts[200]="path_input_counts";
char path_simulated_dir[200]="path_simulated_dir";
char mode[10]="mode";
int max_count = 400;
double step = 1.0;
double h = 4.0; // bandwidth for kde
double max_time = 10.0; // 10 hours
int batch_size = 1800000;
int genes_per_batch = 10000;
double lower_limit = -5.0; // lower limit for parameters
double upper_limit = 2.0; // upper limit for parameters
double k_deg = -1.0;

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

tuple <fs::path, fs::path> run_path_checks(fs::path path_indir){
	// check to see if output_dir exists
	
	fs::path filename_output_profiles ("profile_likelihoods.csv");
	
	fs::path path_output_profiles = path_indir / filename_output_profiles;
	
	fs::path filename_input_parameters ("initial_parameters.csv");
	fs::path path_input_parameters = path_indir / filename_input_parameters;
	
	return make_tuple(path_input_parameters, path_output_profiles);
	
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
	double sum_total = 0.0;
	
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
		distributions[i_dist] = p * sum;
		sum_total += p * sum;
	}
	// normalize
	for(int i_x = 0; i_x < Nx; ++i_x){
		int i_dist = i_param_combination * max_count + i_x;
		distributions[i_dist] /= sum_total;
	}
};

__global__
void generate_kde_gpu_parallel(double *distributions, int *mrna_counts, int max_count, double h, int num_genes, int num_cells)
{
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
		
	for (int i_gene = index; i_gene < num_genes; i_gene+=stride) {
		
		generate_kde_gpu(distributions, mrna_counts, max_count, h, num_genes, i_gene, num_cells);
		
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
		if (strcmp(argv[i], "-i") == 0){
			strcpy(path_input_counts, argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-p") == 0){
			strcpy(path_simulated_dir, argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-gpb") == 0){
			genes_per_batch=atoi(argv[i+1]);
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
void simulate(double max_time, int num_cells, int num_genes, int genes_per_batch, int num_combinations_per_gene, int i_batch, int batch_size, int num_combinations_current_batch, const int num_params, int max_count, double h, double *param_combinations, int *transcriptional_states, int *mrna_count, double *simulated_distributions, double *real_distributions, double *log_likelihoods, curandState* globalState){
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
		
	for (int i_param_combination = index; i_param_combination < num_combinations_current_batch; i_param_combination+=stride) {
		
		if (i_param_combination % 100000 == 0){
			printf("processing batch combo %i...\n", i_param_combination);
		}
		
		// x = i % width;
		// y = (i / width)%height;
		// z = i / (width*height);
		
		int i_gene_in_batch = i_param_combination / num_combinations_per_gene;
		
		int i_gene = i_batch * genes_per_batch + i_gene_in_batch;
		int i_param_combination_full = i_batch * batch_size + i_param_combination;
		
		//printf("%i,%i,%i\n", i_gene_in_batch, i_gene, i_param_combination_full);
		
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
						} else if (i_event == 2){
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
		
		generate_kde_gpu(simulated_distributions, mrna_count, max_count, h, batch_size, i_param_combination, num_cells);
		
		double log_likelihood = 0.0;
		// cycle through counts of REAL distribution!
		// so kde should be generated
		
		for (int i_count = 0; i_count < max_count; i_count++) {
			
			int i_real_count = i_gene * max_count + i_count;
			int i_simulated_count = i_param_combination * max_count + i_count;
			
			int count = (int)(real_distributions[i_real_count] * num_cells);
			
			if (count > 0){
				log_likelihood += log(simulated_distributions[i_simulated_count]) * count;
			}
		}
		log_likelihoods[i_param_combination_full] = log_likelihood;
	}
}

vector<string> get_common_elements (vector<string> param_gene_names, vector<string> count_gene_names) {

	vector<string> param_gene_names_sorted;
	vector<string> count_gene_names_sorted;

	param_gene_names_sorted = param_gene_names;
	count_gene_names_sorted = count_gene_names;

	sort(param_gene_names_sorted.begin(), param_gene_names_sorted.end());
	sort(count_gene_names_sorted.begin(), count_gene_names_sorted.end());

	vector<string> common;
	int i = 0;
	int j = 0;

	while (i < param_gene_names_sorted.size() && j < count_gene_names_sorted.size()){
		if (count_gene_names_sorted[i] == param_gene_names_sorted[j]){
			common.push_back(param_gene_names_sorted[i]);
			i += 1;
			j += 1;
		}
		else if (param_gene_names_sorted[i] < count_gene_names_sorted[j]) {
			i += 1;
		}
		else {
			j += 1;
		}
	}
	return common;
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

double get_parameter_value(string path_simulated_dir_string, string regex_string, string name){
	
	std::smatch match;
	std::regex regexp(regex_string);
	
	if (regex_search(path_simulated_dir_string, match, regexp)){
		string match_string = match.str();
		boost::replace_all(match_string, name, "");
		printf("%s: %s\n", name.c_str(), match_string.c_str());
		double value = stof(match_string);
		return(value);
	}
	else {
		printf("%s not found\n", name.c_str());
		exit(1);
	}
	
}

// 3D to 1D and reverse
// x = i % width;
// y = (i / width)%height;
// z = i / (width*height);
// i = x + width*y + width*height*z;

// nvcc /home/data/nlaszik/cuda_simulation/code/cuda/profile_likelihood.cu -o /home/data/nlaszik/cuda_simulation/code/cuda/build/profile_likelihood -lcurand -lboost_filesystem -lboost_system -lineinfo

// SRP299892
// /home/data/nlaszik/cuda_simulation/code/cuda/build/profile_likelihood -gpb 10000 -p /home/data/nlaszik/cuda_simulation/output/SRP299892_ll/no_np/gran1_max400_time10_step0.1_h0.5_lower-3_upper3_deg0 -i /home/data/Shared/shared_datasets/sc_rna_seq/data/SRP299892/seurat/transcript_counts/srr13336770_transcript_counts.filtered.norm.csv

// TEST
// /home/data/nlaszik/cuda_simulation/code/cuda/build/profile_likelihood -gpb 10000 -p /home/data/nlaszik/cuda_simulation/output/test/no_np/gran1_max400_time10_step0.1_h0.5_lower-3_upper3_deg0 -i /home/data/nlaszik/cuda_simulation/input/test_input.csv

int main(int argc, char** argv)
{
	
	if(!parseCommand(argc, argv)) {
		cout<<"Error in arguments..\n";
		exit(0);
	}
	
	// get simulation data from
	fs::path path_indir (path_simulated_dir);
	fs::path input_directory_name;
	if (!dirExists(path_simulated_dir)){
		printf("please provide a valid input directory\n");
		exit(0);
	}
	else {
		input_directory_name = path_indir.filename();
	}
	
	vector<string> regex_strings{"max\\d+(?=_)", "time\\d+\\.?\\d*(?=_)", "step\\d+\\.?\\d*(?=_)", "h\\d+\\.?\\d*(?=_)", "lower-*?\\d+\\.?\\d*(?=_)", "upper-*?\\d+\\.?\\d*(?=_)", "deg\\d+\\.?\\d*"};
	vector<string> names{"max", "time", "step", "h", "lower", "upper", "deg"};
	double max_count_double;
	vector<double*> values{&max_count_double, &max_time, &step, &h, &lower_limit, &upper_limit, &k_deg};
	
	for (int i = 0; i < regex_strings.size(); ++i)
	{
		*values[i] = get_parameter_value(input_directory_name.c_str(), regex_strings[i], names[i]);
	}
	max_count = (int)max_count_double;
	
	// check directories
	fs::path path_mode = path_indir.parent_path().filename();
	string mode = path_mode.string();
	fs::path path_input_parameters;
	fs::path path_output_profiles;
	tie(path_input_parameters, path_output_profiles) = run_path_checks(path_indir);
	
	printf("mode: %s\n", mode.c_str());
	printf("max count: %i\n", max_count);
	printf("max time in seconds: %f\n", max_time);
	printf("genes per batch: %i\n", genes_per_batch);
	printf("step size: %f\n", step);
	printf("h: %f\n", h);
	
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

	// get params so we can write them in for genes
	// can have different number of genes in params file vs. counts file so we need to match them up
	// create a dict with 
	printf("reading parameters file...\n");
	ifstream paramsfile(path_input_parameters.c_str());
	vector<string> rows_params;
	std::string line_params;
	vector<string> param_gene_names;

	int *params;
	const int num_params = 6;
	
	while (getline(paramsfile, line_params, '\n'))
	{
		rows_params.push_back(line_params); //Get each line of the file as a string
	}
	
	// first we just get names and order of genes
	std::size_t last_pos = 0;
	std::size_t pos = 0;
	for (int i=1; i<rows_params.size(); ++i){
		// the first thing will be a string gene_id
		last_pos = (size_t)-1;
		pos = rows_params[i].find(",", last_pos + 1);
		param_gene_names.push_back(rows_params[i].substr(last_pos + 1, pos - last_pos - 1));
	}
	
	// load real distributions from cell count matrix
	// also make initial guesses for parameters
	ifstream infile(path_input_counts);
	vector<string> rows;
	vector<string> count_gene_names;
	std::string line;
	
	printf("reading input counts...\n");
	while (getline(infile, line, '\n'))
	{
		rows.push_back(line); //Get each line of the file as a string
	}
	int num_cells = rows.size() - 1;
	
	int count_num_genes = 0;
	// process first row of real counts... rows are gene names first column will be "cell" or empty... this handles them both
	last_pos = 0;
	pos = rows[0].find(",", last_pos);
	last_pos = pos;
	while (pos != std::string::npos){
		pos = rows[0].find(",", last_pos + 1);
		count_gene_names.push_back(rows[0].substr(last_pos + 1, pos - last_pos - 1));
		last_pos = pos;
		count_num_genes += 1;
	}

	// find common genes between the two inputs
	vector<string> gene_names = get_common_elements(param_gene_names, count_gene_names);

	printf("number of count genes: %d\n", count_num_genes);
	printf("number of param genes: %lu\n", param_gene_names.size());
	printf("number of common genes: %lu\n", gene_names.size());
	
	int num_genes = gene_names.size();
	cudaMallocManaged(&params, num_genes * num_params * sizeof(int));

	// we now have a list of common genes
	// for each parameter set we need a matching real kde so everything has to be in the same order
	// probably the best way to do this is to use sorting kind of like with the common elements
	// iterate through the original list, find the entry in the sorted list, will be O(m*n)
	// at this point we have a mapping we can use... then
	
	map<int, int> map_param;
	for (int i_gene = 0; i_gene < param_gene_names.size(); i_gene++) {
		string gene_name = param_gene_names[i_gene];
		std::vector<string>::iterator itr = std::find(gene_names.begin(), gene_names.end(), gene_name);
		if(itr != gene_names.end()) {
			int i_common = std::distance(gene_names.begin(), itr);
			map_param[i_gene] = i_common;
		}
	}
	
	map<int, int> map_counts;
	for (int i_gene = 0; i_gene < count_gene_names.size(); i_gene++) {
		string gene_name = count_gene_names[i_gene];
		std::vector<string>::iterator itr = std::find(gene_names.begin(), gene_names.end(), gene_name);
		if(itr != gene_names.end()) {
			int i_common = std::distance(gene_names.begin(), itr);
			map_counts[i_gene] = i_common;
		}
	}
	
	printf("number of cells: %i\n", num_cells);
	printf("number of genes: %i\n", num_genes);
	
	int *real_mrna_count;
	double *real_distributions, *simulated_distributions, *log_likelihoods;
	
	cudaMallocManaged(&real_distributions, num_genes * max_count * sizeof(double));
	cudaMallocManaged(&real_mrna_count, num_genes * num_cells * sizeof(int));
	printf("able to initialize memory on gpu...\n");
	
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
			
			// determine if this gene should be included... and where it should go
			map<int, int>::iterator it = map_counts.find(i_gene);
			if (map_counts.end() != it){
				int i_gene_real = map_counts[i_gene];
				int i_cell_gene = i_cell * num_genes + i_gene_real;
				real_mrna_count[i_cell_gene] = count;
			}
			
			last_pos = pos;
			i_gene++;
		}
	}
	
	int N_init = num_genes;
	int blockSize_init = 32;
	int numBlocks_init = (N_init + blockSize_init - 1) / blockSize_init;
	generate_kde_gpu_parallel<<<numBlocks_init, blockSize_init>>>(real_distributions, real_mrna_count, max_count, h, num_genes, num_cells);
	
	vector<vector<double>> params_vector(num_genes);
	last_pos = 0;
	pos = 0;
	for (int i=1; i<rows_params.size(); ++i){
		// the first thing will be a string gene_id
		int i_gene = i - 1;
		map<int, int>::iterator it = map_param.find(i_gene);
		if (map_param.end() != it){
			
			int i_gene_real = map_param[i_gene];
			
			last_pos = (size_t)-1;
			pos = rows_params[i].find(",", last_pos + 1);
			last_pos = pos;
			
			int i_param = 0;
			float parameter = 0;
			while (pos != std::string::npos){
				pos = rows_params[i].find(",", last_pos + 1);
				if (rows_params[i].substr(last_pos + 1, pos - last_pos - 1).empty()) {
					parameter = 0.0;
				}
				else {
					parameter = stof(rows_params[i].substr(last_pos + 1, pos - last_pos - 1));
					int i_param_gene = i_gene_real * num_params + i_param;
					params[i_param_gene] = parameter;
					params_vector[i_gene_real].push_back(parameter);
					i_param++;
				}
				// initialize param values
				last_pos = pos;
			}
		}
	}

	
	int *transcriptional_states, *mrna_count;
	double *param_combinations;
	
	// creating parameter combinations
	printf("creating parameter combinations...\n");
	
	// these are rates / second 
	// max rate should be once every 5 seconds = 720.0/hour = 0.2/sec... for high range, maybe instead just do linear rate changes 0.195, 0.19, 0.185, ... etc
	// for low range, next is maybe 0.19, 0.18, 0.1
	
	// for each parameter set
	
	// since log range, we start with negatives
	
	double param_lower_limits[num_params] = {lower_limit, 		lower_limit, 	lower_limit, 	lower_limit,	0.0, k_deg};
	double param_upper_limits[num_params] = {upper_limit, 		upper_limit, 	upper_limit, 	upper_limit,	1.0, k_deg};
	
	if (strcmp(mode.c_str(), "no_np") == 0){
		param_lower_limits[3] = -DBL_MAX;
		param_upper_limits[3] = -DBL_MAX;
		param_lower_limits[4] = 0.0;
		param_upper_limits[4] = 0.0;
	}
	else if (strcmp(mode.c_str(), "const") == 0){
		param_lower_limits[0] = DBL_MAX;
		param_upper_limits[0] = DBL_MAX;
		param_lower_limits[1] = -DBL_MAX;
		param_upper_limits[1] = -DBL_MAX;
		param_lower_limits[3] = -DBL_MAX;
		param_upper_limits[3] = -DBL_MAX;
		param_lower_limits[4] = 0.0;
		param_upper_limits[4] = 0.0;
	}
	else if (strcmp(mode.c_str(), "no_knp") == 0){
		param_lower_limits[3] = -DBL_MAX;
		param_upper_limits[3] = -DBL_MAX;
	}
	else if (strcmp(mode.c_str(), "no_pnp") == 0){
		param_lower_limits[4] = 0.0;
		param_upper_limits[4] = 0.0;
	}
	else if (strcmp(mode.c_str(), "full_model") == 0){
		printf("All parameters selected.\n");
	}
	else {
		printf("Invalid mode provided. Please designate as 'const', 'no_np', 'no_knp', 'no_pnp', or 'full_model'\n");
		exit(0);
	}
	
	double step_sizes[num_params] = {(double)(step), (double)(step), (double)(step), (double)(step), (double)(step), (double)(step)};
	
	// each gene should have range/step + range/step + range/step number of combinations of things to try
	// we could parallelize over genes since that isn't too many things to try for profile honestly
	// we could also do profile likelihood though if we want to decrease step... would require ... 3 dimensional indices I think
	
	// data requirements:
	// for each gene, for each parameter, need a distribution along the parameter, length of distribution array is dictated by maximum num_steps I think? since it should be square
	// so yeah it's a 3d thing... ugh... it actually comes out to greater than 2 mil combinations for the entire dataset so we probably want to batch it and parallelize over combinations
	
	// find max combination
	
	vector<vector<double>> param_combinations_vector;
	// create parameters combinations... 
	// we go through all the genes
	int i_combination = 0;
	for (int i_gene = 0; i_gene < num_genes; i_gene++){
		for (int i_param = 0; i_param < num_params; i_param++){
			if (param_lower_limits[i_param] != param_upper_limits[i_param]){
				
				for (double param = param_lower_limits[i_param]; param <= param_upper_limits[i_param]; param += step_sizes[i_param]){
					// this is the dummy starting point
					param_combinations_vector.push_back(params_vector[i_gene]);
					
					if (i_param == 4){
						// p_np is a probability, not a rate
						param_combinations_vector[i_combination][i_param] = param;
					}
					else {
						// rate, so use log
						param_combinations_vector[i_combination][i_param] = pow(10.0, param);
					}
					i_combination++;
				}
			}
		}
	}
	
	int num_param_combinations = i_combination;
	int num_combinations_per_gene = num_param_combinations / num_genes;
	
	int num_batches;
	if (num_genes < genes_per_batch){
		genes_per_batch = num_genes;
		num_batches = 1;
	} else {
		num_batches = (int)ceil(num_genes / genes_per_batch) + 1;
	}
	
	batch_size = genes_per_batch * num_combinations_per_gene;
	
	printf("num combinations: %i, num comb per gene: %i\n", num_param_combinations, num_combinations_per_gene);
	
	printf("num batches: %i, final batch size: %i, num genes in batch: %i\n", num_batches, batch_size, genes_per_batch);
	
	cudaMallocManaged(&transcriptional_states, batch_size * num_cells * sizeof(int));
	cudaMallocManaged(&mrna_count, batch_size * num_cells * sizeof(int));
	cudaMallocManaged(&simulated_distributions, batch_size * max_count * sizeof(double));
	cudaMallocManaged(&param_combinations, batch_size * num_params * sizeof(double));
	cudaMallocManaged(&log_likelihoods, num_param_combinations * sizeof(double));
	
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
			if (i_batch_combination < num_param_combinations) {
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
		
		simulate<<<numBlocks, blockSize>>>(max_time, num_cells, num_genes, genes_per_batch, num_combinations_per_gene, i_batch, batch_size, i_param_combination, num_params, max_count, h, param_combinations, transcriptional_states, mrna_count, simulated_distributions, real_distributions, log_likelihoods, devStates);
		
		cudaEventRecord(stop);
		cudaDeviceSynchronize();
		
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		
		printf("Elapsed seconds: %f\n", milliseconds/1000);
		
	}
	
	// need to output parameter combinations file
	
	// need to output ll for each combination
	FILE *outfile_profiles;
	outfile_profiles = fopen(path_output_profiles.c_str(), "w");//create a file
	
	printf("writing %s\n", path_output_profiles.c_str());
	fprintf(outfile_profiles, "gene,on,off,tx,np,p_np,deg,log_likelihood\n");
	for (int i_param_combination = 0; i_param_combination < num_param_combinations; i_param_combination++){
		int i_gene = i_param_combination / num_combinations_per_gene;
		fprintf(outfile_profiles, "%s,", gene_names[i_gene].c_str());
		for (int i_param = 0; i_param < num_params; i_param++){
			fprintf(outfile_profiles, "%.16f,", param_combinations_vector[i_param_combination][i_param]);
		}
		fprintf(outfile_profiles, "%.16f\n", log_likelihoods[i_param_combination]);
	}
	fclose(outfile_profiles);

	// Free memory
	param_combinations_vector.clear();
	cudaFree(param_combinations);
	cudaFree(simulated_distributions);
	cudaFree(transcriptional_states);
	cudaFree(mrna_count);
	cudaFree(real_distributions);
	cudaFree(log_likelihoods);
	cudaFree(real_mrna_count);
	cudaFree(devStates);
	cudaFree(d_y);
	return 0;
}




