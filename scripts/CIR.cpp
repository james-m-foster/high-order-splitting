#include <fstream>
#include <iomanip>
#include <random>
#include <chrono>

using namespace std;

// Numerical simulation of the Cox-Ingersoll-Ross (CIR) model on [0, T]:
// dy_t = a(b-y_t) dt + sigma sqrt(y_t) dW_t

// The below class contains the five numerical methods considered in the paper:
//
// J. Foster, G. Dos Reis and C. Strange, High order splitting methods for SDEs
// satisfying a commutativity condition, arxiv.org/abs/2210.17543, 2022.

class CIRmethods {
    public:
        // Input parameters
        double a, b, sigma, T;
        int no_of_steps;

        CIRmethods(double, double, double, double, int);

        double euler_maruyama(double, double);
        double milstein(double, double);
        double drift_implicit_euler(double, double);
        double ninomiya_victoir_splitting(double, double);
        double high_order_splitting(double, double, double);
        double mean(double);
        double variance(double);

    private:
        // Precomputed values that depend on the input parameters
        double half_sigma, sigma_squared, step_size;
        double ab, tilde_b, ah, half_a;
        double first_exp_drift, middle_exp_drift, half_exp_drift;

        // Precomputed constants
        const double pi = 4.0*atan(1.0);
        const double sqrt_pi = sqrt(pi);
        const double first_h = 0.5 - sqrt(3.0)/6.0;
        const double middle_h = sqrt(3.0)/3.0;
        const double root_three = sqrt(3.0);
        const double quarter_plus_root_forty_eight = ((3.0 + sqrt(3.0))/12.0);
        const double quarter_one_minus_root_three = 0.25*(1.0 - sqrt(3.0));
        const double k3 = ((3.0 + sqrt(3.0))/6.0);
        const double k4 = ((3.0 + sqrt(3.0))/3.0);
};

// Constructor will initialize the above variables
CIRmethods::CIRmethods(double input_a, double input_b, double input_sigma, double input_T, int input_no_of_steps){
    a = input_a;
    b = input_b;
    sigma = input_sigma;

    half_sigma = 0.5*sigma;
    sigma_squared = pow(sigma, 2);
    ab = a*b;
    half_a = 0.5*a;

    step_size =  input_T/(double)input_no_of_steps;

    T = input_T;
    no_of_steps = input_no_of_steps;
    step_size =  input_T/(double)input_no_of_steps;

    // tilde_b is obtained from the Ito-Stratonovich correction
    tilde_b = b - a / (4.0*sigma_squared);
    ah = a*step_size;

    // the variables are used in the high order splitting
    first_exp_drift = exp(-(0.5 - sqrt(3.0)/6.0)*a*step_size);
    middle_exp_drift = exp(-(sqrt(3.0)/3.0)*a*step_size);
    half_exp_drift = exp(-0.5*a*step_size);
}

// Compute one step of the (non-negative) Euler-Maruyama method
double CIRmethods::euler_maruyama(double y0, double brownian_increment){
    return max(0.0, y0 + a*(b - y0)*step_size + sigma*sqrt(y0)*brownian_increment);
};

// Compute one step of the (non-negative) Milstein method
double CIRmethods::milstein(double y0, double brownian_increment){

    return max(0.0, ah*(tilde_b - y0) + pow(sqrt(y0) + half_sigma*brownian_increment, 2));
};

// Compute one step of the drift-implicit Euler method from Alfonsi (2005)
double CIRmethods::drift_implicit_euler(double y0, double brownian_increment){
    double y0_plus_increment = sqrt(y0) + half_sigma*brownian_increment;
    double c = 2.0 + a*step_size;

    return pow((y0_plus_increment + sqrt(pow(y0_plus_increment, 2) + c*(ab - 0.25*sigma_squared)*step_size))/c, 2);
};

// Compute one step of the Ninomiya-Victoir scheme (for CIR, it coincides with the Strang splitting)
double CIRmethods::ninomiya_victoir_splitting(double y0, double brownian_increment){
    double y1 = half_exp_drift*y0 + tilde_b*(1.0 - half_exp_drift);
    double y2 = pow(sqrt(y1) + half_sigma*brownian_increment, 2);
    return half_exp_drift*y2 + tilde_b*(1.0 - half_exp_drift);
};

// Compute one step of the high order splitting scheme from Foster et al. (2022)
double CIRmethods::high_order_splitting(double y0, double brownian_increment, double brownian_area){
    double y1 = first_exp_drift*y0 + tilde_b*(1.0 - first_exp_drift);
    double y2 = pow(sqrt(y1) + half_sigma*(0.5*brownian_increment + root_three*brownian_area), 2);
    double y3 = middle_exp_drift*y2 + tilde_b*(1.0 - middle_exp_drift);
    double y4 = pow(sqrt(y3) + half_sigma*(0.5*brownian_increment - root_three*brownian_area), 2);

    return first_exp_drift*y4 + tilde_b*(1.0 - first_exp_drift);
};

// Compute the mean at time T of the CIR model
double CIRmethods::mean(double y0){
    return b + (y0 - b)*exp(-a*T);
};

// Compute the variance at time T of the CIR model
double CIRmethods::variance(double y0){
    return pow(sigma, 2)*((y0 / a)*(exp(-a*T) - exp(-2.0*a*T)) + (b/(2.0*a))*pow((1.0-exp(-a*T)), 2));
};


int main()
{
    // Input parameters
    const double a = 1.0;
    const double b = 1.0;
    const double sigma = 1.0;
    const double y0 = 1.0;
    const double T = 1.0;
    const int no_of_steps = 100;

    // Number of steps used by the "fine" approximation
    // during each step of the "crude" numerical method
    const int no_of_fine_steps = 10;

    // Number of paths used for the Monte Carlo estimators
    const int no_of_paths = 100000;

    // Variance for generating the "space-time" Lévy areas
    const double twelve = 1.0/12.0;

    // Step size parameters
    const double step_size =  T/(double)no_of_steps;
    const double one_over_step_size = 1.0/step_size;
    const double fine_step_size = T/(double)(no_of_steps*no_of_fine_steps);

    // We will be comparing the methods on two different time scales
    CIRmethods course_method(a, b, sigma, T, no_of_steps);
    CIRmethods fine_method(a, b, sigma, T, no_of_steps*no_of_fine_steps);

    // Normal distributions for generating the Brownian increments and space-time Lévy areas
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> increment_distribution(0.0,sqrt(step_size));
    std::normal_distribution<double> area_distribution(0.0, sqrt(twelve*step_size));
    std::normal_distribution<double> fine_increment_distribution(0.0,sqrt(fine_step_size));
    std::normal_distribution<double> fine_area_distribution(0.0, sqrt(twelve*fine_step_size));

    // Numerical solutions computed with course and fine step sizes
    double y_1 = y0;
    double y_2 = y0;
    double y_fine = y0;

    // Information about the Brownian motion (increments and space-time Lévy areas)
    // Using the notation in Foster et al. (2022), these are
    // brownian_increment is W_{s,t}
    // brownian_area      is H_{s,t}
    double brownian_increment = 0.0;
    double brownian_area = 0.0;
    double fine_brownian_increment = 0.0;
    double fine_brownian_area = 0.0;

    // Strong error estimators for y at time T
    double error_1 = 0.0;
    double error_2 = 0.0;

    for (int i=0; i<no_of_paths; ++i) {
        for (int j=1; j<=no_of_steps; ++j) {

            brownian_increment = 0.0;
            brownian_area = 0.0;

            for (int k=1; k<= no_of_fine_steps; ++k){
                // Generate information about the Brownian path over the "fine" increment
                fine_brownian_increment = fine_increment_distribution(generator);
                fine_brownian_area = fine_area_distribution(generator);

                // Propagate the numerical solution over the fine increment
                y_fine = fine_method.high_order_splitting(y_fine, fine_brownian_increment, fine_brownian_area);

                // Update the information about the Brownian path over the
                // course interval using the recently generated variables.
                // The below procedure can be derived using some elementary
                // properties of integration (additivity and linearity)
                brownian_area = brownian_area + fine_step_size
                                 * (brownian_increment + 0.5*fine_brownian_increment
                                                       + fine_brownian_area);

                // Compute the increment of the Brownian motion over the course interval
                brownian_increment = brownian_increment + fine_brownian_increment;
            }

            // Compute the space-time area of the path over the course interval
            brownian_area = (brownian_area - 0.5*step_size*brownian_increment)*one_over_step_size;

            // Propagate the numerical solutions over the course interval
            y_1 = course_method.ninomiya_victoir_splitting(y_1, brownian_increment);
            y_2 = course_method.high_order_splitting(y_2, brownian_increment, brownian_area);
        }

        // Compute the L2 error between the methods on the fine and course scales
        error_1 = error_1 + pow(y_1 - y_fine, 2);
        error_2 = error_2 + pow(y_2 - y_fine, 2);

        // Reset the numerical solutions
        y_1 = y0;
        y_2 = y0;
        y_fine = y0;
    }

    double double_no_of_paths = double(no_of_paths);

    error_1 = sqrt(error_1/double_no_of_paths);
    error_2 = sqrt(error_2/double_no_of_paths);

    double y_test = y0;

    // Time the numerical method
    auto start = std::chrono::high_resolution_clock::now();

    for (int i=0; i<no_of_paths; ++i) {
        for (int j=1; j<=no_of_steps; ++j) {

            // Generate information about Brownian path
            brownian_increment = increment_distribution(generator);
            // brownian_area = area_distribution(generator);

            // Propagate the numerical solution over the course interval
            y_test = course_method.euler_maruyama(y_test, brownian_increment);
        }

        // Reset the numerical solution
        y_test = y0;
    }

    auto finish = std::chrono::high_resolution_clock::now();

    // Obtain the time taken by the speed test
    std::chrono::duration<double> elapsed = finish - start;


    // Display the results in a text file
    ofstream myfile;
    myfile.open ("cir_sim.txt");

    myfile << std::fixed << std::setprecision(15) \
           << "Number of steps: " << "\t\t\t" << no_of_steps << "\n";

    myfile << std::fixed << std::setprecision(10) \
           << "Number of sample paths: " << "\t\t" << no_of_paths << "\n\n";

    myfile << std::fixed << std::setprecision(2) << "L2 error at time T = " << T \
           << " for method 1: " << std::setprecision(15) << error_1 << "\n" \
           << "L2 error at time T = " << std::setprecision(2) << T \
           << " for method 2: " << std::setprecision(15) << error_2 << "\n\n";

    myfile << std::fixed << std::setprecision(10) \
           << "Time taken in speed test: " << "\t\t" << elapsed.count();

    myfile.close();

    return 0;
}