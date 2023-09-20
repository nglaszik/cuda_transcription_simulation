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
#include <regex>

#include <sys/types.h>
#include <sys/stat.h>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string/replace.hpp>

using namespace std;
namespace fs = boost::filesystem;

char path_output_dir[200]="path_output_dir";
char path_real_counts[200]="path_real_counts";
char path_simulated_dir[200]="path_simulated_dir";
int batch_size = 0;

int max_count = 100;
double step = 1.0;
double h = 4.0; // bandwidth for kde
double max_time = 3600.0; // 1 hour
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

tuple <fs::path, fs::path, fs::path, fs::path, fs::path, fs::path> run_path_checks(fs::path path_indir, fs::path path_outdir, int max_count, float max_time, float step, float h, float lower_limit, float upper_limit, float k_deg, fs::path mode_dir){
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
	
	fs::path filename_counts ("initial_counts.csv");
	fs::path filename_parameters ("initial_parameters.csv");
	fs::path filename_mses ("initial_mses.csv");
	
	fs::path path_output_counts = path_run_dir / filename_counts;
	fs::path path_output_parameters = path_run_dir / filename_parameters;
	fs::path path_output_mses = path_run_dir / filename_mses;
	
	fs::path filename_simulated_counts ("counts.csv");
	fs::path filename_simulated_parameters ("parameters.csv");
	fs::path filename_simulated_kdes ("kdes.bin");
	
	fs::path path_simulated_counts = path_indir / filename_simulated_counts;
	fs::path path_simulated_parameters = path_indir / filename_simulated_parameters;
	fs::path path_simulated_kdes = path_indir / filename_simulated_kdes;
	
	return make_tuple(path_output_counts, path_output_parameters, path_output_mses, path_simulated_counts, path_simulated_parameters, path_simulated_kdes);
	
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
        if (strcmp(argv[i], "-o") == 0){
			strcpy(path_output_dir, argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-i") == 0){
			strcpy(path_real_counts, argv[i+1]);
			i=i+2;
		}
		else if (strcmp(argv[i], "-d") == 0){
			strcpy(path_simulated_dir, argv[i+1]);
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
void find_initial_best_fit(int num_genes, int batch_size, int num_combinations_in_batch, int i_batch, int max_count, double *simulated_distributions, double *real_distributions, double *best_mses, int *best_params){
	
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int i_gene = index; i_gene < num_genes; i_gene+=stride) {
		for (int i_param_combination = 0; i_param_combination < num_combinations_in_batch; i_param_combination++) {
			//if (i_param_combination % 100000 == 0 && i_gene == 0) printf("processing param combination %i\n", i_param_combination);
			double mse = 0.0;
			for (int i_count = 0; i_count < max_count; i_count++ ) {
				int i_real_count = i_gene * max_count + i_count;
				int i_initial_count = i_param_combination * max_count + i_count;
				mse += (simulated_distributions[i_initial_count] - real_distributions[i_real_count]) * (simulated_distributions[i_initial_count] - real_distributions[i_real_count]);
			}
			
			mse = mse / (double)max_count;
			if (mse < best_mses[i_gene]){
				best_mses[i_gene] = mse;
				best_params[i_gene] = i_batch * batch_size + i_param_combination;
				//printf("mse: %.16f, gene %i\n", mse, i_gene);
			}
		}
	}
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
		exit(0);
	}
	
}

// 3D to 1D and reverse
// x = i % width;
// y = (i / width)%height;
// z = i / (width*height);
//i = x + width*y + width*height*z;

// nvcc /home/data/nlaszik/cuda_simulation/code/cuda/fit_distributions_mse.cu -o /home/data/nlaszik/cuda_simulation/code/cuda/build/fit_distributions_mse -lcurand -lboost_filesystem -lboost_system -lineinfo

// /home/data/nlaszik/cuda_simulation/code/cuda/build/fit_distributions_mse -bs 1000000 -i /home/data/Shared/shared_datasets/sc_rna_seq/data/SRP299892/seurat/transcript_counts/srr13336770_transcript_counts.filtered.norm.csv -d /home/data/nlaszik/cuda_simulation/output/simulated/no_np/max400_time1000_step0.1_h2_lower-3_upper3_deg0 -o /home/data/nlaszik/cuda_simulation/output/SRP299892

// /home/data/nlaszik/cuda_simulation/code/cuda/build/fit_distributions_mse -bs 1000000 -i /home/data/Shared/shared_datasets/sc_rna_seq/data/SRP299892/seurat/transcript_counts/srr13336770_transcript_counts.filtered.norm.csv -d /home/data/nlaszik/cuda_simulation/output/simulated_methylation/k_on/max400_time10_step0.2_h2_lower-3_upper3_deg0 -o /home/data/nlaszik/cuda_simulation/output/SRP299892

// /home/data/nlaszik/cuda_simulation/code/cuda/build/fit_distributions_mse -bs 1000000 -i /home/data/Shared/shared_datasets/sc_rna_seq/data/SRP299892/seurat/transcript_counts/srr13336770_transcript_counts.filtered.norm.csv -d /home/data/nlaszik/cuda_simulation/output/simulated_methylation_test/k_tx/max400_time10_step0.15_h2_lower-3_upper3_deg0 -o /home/data/nlaszik/cuda_simulation/output/SRP299892

// /home/data/nlaszik/cuda_simulation/code/cuda/build/fit_distributions_mse -bs 1000000 -i /home/data/Shared/shared_datasets/sc_rna_seq/nlaszik/dko_atrinh_102022/seurat/transcript_counts/rep2_transcript_counts.filtered.norm.csv -d /home/data/nlaszik/cuda_simulation/output/simulated/no_np/max400_time1000_step0.1_h2_lower-3_upper3_deg0 -o /home/data/nlaszik/cuda_simulation/output/dko_hesc

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
	fs::path path_simulated_counts;
	fs::path path_simulated_params;
	fs::path path_simulated_kdes;
	fs::path path_mode = path_indir.parent_path().filename();
	
	fs::path path_outdir (path_output_dir);
	fs::path path_output_counts;
	fs::path path_output_parameters;
	fs::path path_output_mses;
	
	tie(path_output_counts, path_output_parameters, path_output_mses, path_simulated_counts, path_simulated_params, path_simulated_kdes) = run_path_checks(path_indir, path_outdir, max_count, max_time, step, h, lower_limit, upper_limit, k_deg, path_mode);
	
	int num_cells = 0;
	int num_genes = 0;
	int i_cell_gene = 0;
	int num_param_combinations = 0;
	int num_params = 0;
	
	printf("mode: %s\n", path_mode.c_str());
	printf("max count: %i\n", max_count);
	printf("batch size: %i\n", batch_size);
	
	float free_m,total_m,used_m;

	size_t free_t,total_t;
	
	cudaMemGetInfo(&free_t,&total_t);
	
	free_m =(float)free_t/1048576.0;
	
	total_m=(float)total_t/1048576.0;
	
	used_m=total_m-free_m;
	
	printf ("mem free %f MB, mem total %f MB, mem used %f MB\n", free_m, total_m, used_m);
	
	// shared file-reading vars
	std::size_t last_pos = 0;
	std::size_t pos = 0;
	std::string line;
	
	///////////////////
	// LOAD REAL COUNTS
	///////////////////
	printf("reading real counts %s...\n", path_real_counts);
	ifstream file_real_counts(path_real_counts);
	vector<string> rows_real;
	vector<string> gene_names;
	while (getline(file_real_counts, line, '\n'))
	{
	    rows_real.push_back(line); //Get each line of the file as a string
	}
	// process first row of real counts... first column will be "cell" or empty... this handles them both
	pos = rows_real[0].find(",", last_pos);
	last_pos = pos;
	while (pos != std::string::npos){
		pos = rows_real[0].find(",", last_pos + 1);
		gene_names.push_back(rows_real[0].substr(last_pos + 1, pos - last_pos - 1));
		last_pos = pos;
		num_genes += 1;
	}
	num_cells = rows_real.size() - 1;
	printf("number of real genes: %i\n", num_genes);
	printf("number of real cells: %i\n", num_cells);
	
	///////////////////
	// LOAD PARAMETERS
	///////////////////
	printf("reading parameters file %s...\n", path_simulated_params.c_str());
	ifstream paramsfile(path_simulated_params.c_str());
	vector<string> rows_params;
	std::string line_params;
	
	while (getline(paramsfile, line_params, '\n'))
	{
	    rows_params.push_back(line_params); //Get each line of the file as a string
	}
	num_param_combinations = rows_params.size() - 1;
	printf("number of param combinations: %i\n", num_param_combinations);
	
	string param_header = rows_params[0];
	for (int i_s = 0; i_s < param_header.size(); i_s++) if (param_header[i_s] == ',') num_params++;
	printf("number of params: %i\n", num_params);
	
	double *param_combinations = new double[num_param_combinations * num_params];
	
	for (int i=1; i<rows_params.size(); ++i){
		// the first thing will be a string cell_id
		last_pos = (size_t)-1;
		pos = 0;
		int i_param_combination = i - 1;
		int i_param = 0;
		double parameter = 0.0;
		while (pos != std::string::npos){
			pos = rows_params[i].find(",", last_pos + 1);
			if (rows_params[i].substr(last_pos + 1, pos - last_pos - 1).empty()) {
				parameter = 0.0;
			}
			else {
				parameter = stof(rows_params[i].substr(last_pos + 1, pos - last_pos - 1));
				int i_param_combination_param = i_param_combination * num_params + i_param;
				param_combinations[i_param_combination_param] = parameter;
				i_param++;
			}
			// initialize param values
			last_pos = pos;
		}
	}
	
	int *best_params, *real_mrna_count;
	double *real_distributions, *simulated_distributions, *best_mses;
	
	cudaMallocManaged(&best_mses, num_genes * sizeof(double));
	cudaMallocManaged(&best_params, num_genes * sizeof(double));
	cudaMallocManaged(&real_distributions, num_genes * max_count * sizeof(double));
	cudaMallocManaged(&real_mrna_count, num_genes * num_cells * sizeof(int));
	
	// init best mses
	for (int i_gene = 0; i_gene < num_genes; i_gene++) {
		best_mses[i_gene] = 1.0;
		best_params[i_gene] = 0;
	}
	
	///////////////////
	//GENERATE KDES
	///////////////////	
	printf("generating real count kdes...\n");
	// init real distributions
	for (int i_fill = 0; i_fill < num_genes * max_count; i_fill++) {
		real_distributions[i_fill] = 0.0;
	}
	int s = rows_real.size();
	for (int i=1; i<s; ++i){
		// the first thing will be a string cell_id
		last_pos = 0;
		pos = rows_real[i].find(",", last_pos + 1);
		last_pos = pos;
		
		int i_gene = 0;
		int i_cell = i - 1;
		int count = 0;
		while (pos != std::string::npos){
			pos = rows_real[i].find(",", last_pos + 1);
			if (rows_real[i].substr(last_pos + 1, pos - last_pos - 1).empty()) {
				count = 0;
			}
			else {
				count = stoi(rows_real[i].substr(last_pos + 1, pos - last_pos - 1));
			}
			// initialize cell values
			i_cell_gene = i_cell * num_genes + i_gene;
			real_mrna_count[i_cell_gene] = count;
			last_pos = pos;
			i_gene++;
		}
	}
	
	// set up to fit on gpu
	int N_init = num_genes;
	int blockSize_init = 32;
	int numBlocks_init = (N_init + blockSize_init - 1) / blockSize_init;
	generate_kde_gpu_parallel<<<numBlocks_init, blockSize_init>>>(real_distributions, real_mrna_count, max_count, h, num_genes, num_cells);
	
	int num_batches = (int)ceil(num_param_combinations / batch_size) + 1;
	
	printf("number of param combinations: %i, number of batches: %i\n", num_param_combinations, num_batches);
	
	cudaMallocManaged(&simulated_distributions, batch_size * max_count * sizeof(double));
	double *simulated_distributions_host = new double[batch_size * max_count];
	
	// Run kernel on the GPU
	int N = num_genes;
	int blockSize = 32;
	int numBlocks = (N + blockSize - 1) / blockSize;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	FILE *file_kde;
	file_kde = fopen(path_simulated_kdes.c_str(), "rb");
	
	for (int i_batch = 0; i_batch < num_batches; i_batch++){
		
		printf("processing parameter batch %i, loading kdes into gpu memory...\n", i_batch + 1);
		// read part of file into host memory
		double *pt;
		int i_combination_in_batch = 0;
		for (int i_batch_combination = i_batch * batch_size; i_batch_combination < (i_batch + 1) * batch_size; i_batch_combination++){
			if (i_batch_combination < num_param_combinations){
				double sum_total = 0.0;
				for (int i_count = 0; i_count < max_count; i_count++){
					int i_dist = i_combination_in_batch * max_count + i_count;
					pt = &simulated_distributions_host[i_dist];
					fread(pt, sizeof(double), 1, file_kde);
					simulated_distributions[i_dist] = simulated_distributions_host[i_dist];
					sum_total += simulated_distributions_host[i_dist];
				}
				// normalize
				for(int i_count = 0; i_count < max_count; i_count++){
					int i_dist = i_combination_in_batch * max_count + i_count;
					simulated_distributions[i_dist] /= sum_total;
				}
				i_combination_in_batch++;
			}
			else {
				break;
			}
		}
		
		int num_combinations_in_batch = i_combination_in_batch;
		printf("num combinations in batch: %i\n", num_combinations_in_batch);
		
		find_initial_best_fit<<<numBlocks, blockSize>>>(num_genes, batch_size, num_combinations_in_batch, i_batch, max_count, simulated_distributions, real_distributions, best_mses, best_params);
		
		cudaEventRecord(stop);
		cudaDeviceSynchronize();
		
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("Elapsed seconds: %f\n", milliseconds/1000);
		
	}
	
	fclose(file_kde);
	
	printf("%s\n", path_output_mses.c_str());
	FILE *outfile_mse;
	outfile_mse = fopen(path_output_mses.c_str(), "w");//create a file
	for (int i_gene = 0; i_gene < num_genes; i_gene++){
		fprintf(outfile_mse, "%s,%.16f\n", gene_names[i_gene].c_str(), best_mses[i_gene]);
	}
	fclose(outfile_mse);
	
	//output parameters to csv
	printf("%s\n", path_output_parameters.c_str());
	FILE *outfile_parameters;
	outfile_parameters = fopen(path_output_parameters.c_str(), "w");//create a file
	
	// no need to iterate cells, since each gene in each cell will have the same params
	fprintf(outfile_parameters, "gene,%s\n", param_header.c_str());
    for (int i_gene = 0; i_gene < num_genes; i_gene++){
	    fprintf(outfile_parameters, "%s,", gene_names[i_gene].c_str());
	    for (int i_param = 0; i_param < num_params; i_param++){
		    int i_param_combination_param = best_params[i_gene] * num_params + i_param;
		    fprintf(outfile_parameters, "%.16f,", param_combinations[i_param_combination_param]);
		}
		fprintf(outfile_parameters, "\n");
    }
	fclose(outfile_parameters);
	
	///////////////////
	// LOAD SIMULATED COUNTS
	///////////////////
	FILE *outfile_counts;
	outfile_counts = fopen(path_output_counts.c_str(), "w");//create a file
	
	printf("reading and writing best simulated counts %s...\n", path_simulated_counts.c_str());
	ifstream file_simulated_counts(path_simulated_counts.c_str());
	int i_count_combination = 0;
	while (getline(file_simulated_counts, line, '\n'))
	{
		for (int i_gene = 0; i_gene < num_genes; i_gene++){
			if (best_params[i_gene] == i_count_combination){
				fprintf(outfile_counts, "%s,", gene_names[i_gene].c_str());
				fprintf(outfile_counts, "%s\n", line.c_str());
			}
		}
		i_count_combination++;
	}
	fclose(outfile_counts);
	
	// Free memory
	delete [] param_combinations;
	delete [] simulated_distributions_host;
	cudaFree(real_mrna_count);
	cudaFree(best_mses);
	cudaFree(best_params);
	cudaFree(real_distributions);
	cudaFree(simulated_distributions);
	return 0;
}




