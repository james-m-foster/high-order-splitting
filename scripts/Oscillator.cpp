#include <fstream>
#include <iomanip>
#include <random>
#include <chrono>
#include <cmath>

using namespace std;

// Numerical simulation of the anharmonic oscillator on [0, T]:
// dy_t = sin(y_t) dt + sigma dW_t

// The below class contains the four numerical methods considered in the paper:
//
// J. Foster, G. Dos Reis and C. Strange, High order splitting methods for SDEs
// satisfying a commutativity condition, arxiv.org/abs/2210.17543, 2022.

class additiveSDEmethods {
    public:
        // Input parameters
        double sigma, T;
        int no_of_steps;

        additiveSDEmethods(double, double, int);

        double f(double);
        double euler_maruyama(double, double);
        double shifted_euler(double, double, double);
        double SRA1(double, double, double);
        double shifted_ralston(double, double, double, int);

    private:
        // Precomputed values that depend on the input parameters
        double step_size;
        double sqrt_step_size;
        double k_constant;

        // Precomputed constants
        const double third = 1.0/3.0;
        const double two_thirds = 2.0/3.0;
};

// Constructor will initialize the above variables
additiveSDEmethods::additiveSDEmethods(double input_sigma,
                                   double input_T, int input_no_of_steps){

    T = input_T;
    sigma = input_sigma;
    no_of_steps = input_no_of_steps;

    step_size =  input_T/(double)input_no_of_steps;
    sqrt_step_size =  sqrt(step_size);

    // This constant is required by the high order splitting
    // (see Section 4 or Appendix B of Foster et al. (2022) for details)
    k_constant = (0.125*sqrt_step_size)/sqrt(atan(1.0)*24.0);
};

// Evaluate the vector field
double additiveSDEmethods::f(double y){

    return sin(y);
};

// Compute one step of the Euler-Maruyama method
double additiveSDEmethods::euler_maruyama(double y0, double brownian_increment){

    return y0 + f(y0) * step_size + sigma * brownian_increment;
};

// Compute one step of the Shifted Euler method (Section 5 of Foster et al. (2022))
double additiveSDEmethods::shifted_euler(double y0, double brownian_increment, double brownian_area){

    return y0 + f(y0 + sigma * 0.5 * brownian_increment + brownian_area) * step_size + sigma * brownian_increment;
};

// Compute one step of a two-stage strong 1.5 Stochastic Runge-Kutta scheme from:
// Andreas Rößler, Runge–Kutta Methods for the Strong Approximation of Solutions
// of Stochastic Differential Equations, SIAM Journal on Numerical Analysis, 2010.
double additiveSDEmethods::SRA1(double y0, double brownian_increment,
                                   double brownian_area){
    double fy0 = f(y0);

    double y1 = y0 + 0.75 * (fy0*step_size + sigma * (brownian_increment + 2.0*brownian_area));

    return y0 + (third * fy0 + two_thirds * f(y1)) * step_size + sigma*brownian_increment;
};

// Compute one step of a Shifted Ralston method that employs an optimal integral
// estimator (see Section 5 or Appendix B of Foster et al. (2022) for details)
double additiveSDEmethods::shifted_ralston(double y0, double brownian_increment, double brownian_area, int brownian_swing){

    double expected_k = k_constant*brownian_swing;

    double epsilon = 1.0;

    if (12.0*expected_k < brownian_increment){
        epsilon = -1.0;
    }

    double b_minus_c = epsilon*sqrt(pow(brownian_increment,2) - 24.0*brownian_increment*expected_k + 2.4*pow(brownian_area,2) + 0.8*step_size);

    double b = 0.5*(brownian_increment + 2.0*brownian_area + b_minus_c);

    double y_shift = y0 + sigma * b;

    double f_y_shift= f(y_shift);

    double f_y_two_thirds = f(y_shift + two_thirds*(f_y_shift * step_size - sigma * b_minus_c));

    return y0 + (0.25*f_y_shift + 0.75*f_y_two_thirds)*step_size + sigma * brownian_increment;
};


int main()
{
    // Input parameters
    const double sigma = 1.0;
    const double y0 = 1.0;
    const double T = 1.0;
    const int no_of_steps = 1000;

    // Number of steps used by the "fine" approximation
    // during each step of the "crude" numerical method
    const int no_of_fine_steps = 10;

    // Number of paths used for the Monte Carlo estimators
    const int no_of_paths = 1000;

    // Variance for generating the "space-time" Lévy areas
    const double twelve = 1.0/12.0;

    // Step size parameters
    const double step_size =  T/(double)no_of_steps;
    const double one_over_half_step_size = 2.0/step_size;
    const double fine_step_size = T/(double)(no_of_steps*no_of_fine_steps);

    // We will be comparing the methods on two different time scales
    additiveSDEmethods course_method(sigma, T, no_of_steps);
    additiveSDEmethods fine_method(sigma, T, no_of_steps*no_of_fine_steps);

    // Normal and Rademacher distributions for generating the Brownian increments,
    // space-time Lévy areas and space-time Lévy swings
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> increment_distribution(0.0, sqrt(step_size));
    std::normal_distribution<double> area_distribution(0.0, sqrt(twelve*step_size));
    std::normal_distribution<double> fine_increment_distribution(0.0, sqrt(fine_step_size));
    std::normal_distribution<double> fine_area_distribution(0.0, sqrt(twelve \
                                                                      *fine_step_size));
    std::uniform_int_distribution<int> swing_distribution(0, 1);

    // Numerical solutions computed with course and fine step sizes
    double y_1 = y0;
    double y_2 = y0;
    double y_fine = y0;

    // Information about the Brownian motion (increments and space-time Lévy areas)
    // Using the notation in Foster et al. (2022), these are
    // brownian_increment is W_{s,t}
    // brownian_area      is H_{s,t}
    // brownian_swing     is n_{s,t}
    double brownian_increment = 0.0;
    double brownian_area = 0.0;
    double brownian_increment_1 = 0.0;
    double brownian_increment_2 = 0.0;
    double brownian_area_1 = 0.0;
    double brownian_area_2 = 0.0;
    int brownian_swing = 1;
    double fine_brownian_increment = 0.0;
    double fine_brownian_area = 0.0;
    // int fine_brownian_swing = 1;

    // Strong error estimators for y at time T
    double error_1 = 0.0;
    double error_2 = 0.0;

    for (int i=0; i<no_of_paths; ++i) {
        for (int j=1; j<=no_of_steps; ++j) {

            brownian_increment_1 = 0.0;
            brownian_increment_2 = 0.0;
            brownian_area_1 = 0.0;
            brownian_area_2 = 0.0;

            for (int k=1; k<= no_of_fine_steps; ++k){
                // Generate information about the Brownian path over the "fine" increment
                fine_brownian_increment = fine_increment_distribution(generator);
                fine_brownian_area = fine_area_distribution(generator);

                // fine_brownian_swing = 2*swing_distribution(generator) - 1;

                // Propagate the numerical solution over the fine increment
                y_fine = fine_method.SRA1(y_fine, fine_brownian_increment, fine_brownian_area);

                // Update the information about the Brownian path over the
                // course interval using the recently generated variables.
                // The below procedure can be derived using some elementary
                // properties of integration (additivity and linearity)

                // Since we are using space-time Lévy swings, we shall
                // first do this over the two half-intervals seperately
                if (2*k <= no_of_fine_steps){
                    brownian_area_1 = brownian_area_1 + fine_step_size
                                            * (brownian_increment_1 + 0.5*fine_brownian_increment \
                                                          + fine_brownian_area);

                    // Compute the increment of the Brownian motion over half the course interval
                    brownian_increment_1 = brownian_increment_1 + fine_brownian_increment;
                }
                else
                {
                    brownian_area_2 = brownian_area_2 + fine_step_size
                                            * (brownian_increment_2 + 0.5*fine_brownian_increment \
                                                          + fine_brownian_area);

                    // Compute the increment of the Brownian motion over half the course interval
                    brownian_increment_2 = brownian_increment_2 + fine_brownian_increment;
                }
            }

            // Compute the space-time Lévy areas of the path over the course half-intervals
            brownian_area_1 = brownian_area_1*one_over_half_step_size - 0.5*brownian_increment_1;
            brownian_area_2 = brownian_area_2*one_over_half_step_size - 0.5*brownian_increment_2;

            // As we did these computations for each half-interval, we can compute
            // the space-time Lévy swing (see Foster et al. (2022) for details)
            brownian_swing = 1;

            if (brownian_area_1 < brownian_area_2){
                brownian_swing = -1;
            }

            // Compute the increment of the Brownian motion over the course interval
            brownian_increment = brownian_increment_1 + brownian_increment_2;

            // Compute the space-time area of the path over the course interval
            brownian_area = 0.5*(brownian_area_1 + brownian_area_2) + 0.25*(brownian_increment_1 - brownian_increment_2);

            // Propagate the numerical solutions over the course interval
            y_1 = course_method.SRA1(y_1, brownian_increment, brownian_area);
            y_2 = course_method.shifted_ralston(y_2, brownian_increment, brownian_area, brownian_swing);

            //y_1 = course_method.euler_maruyama(y_1, brownian_increment);
            //y_2 = course_method.shifted_euler(y_2, brownian_increment);
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
            brownian_area = area_distribution(generator);
            // brownian_swing = 2*swing_distribution(generator) - 1;

            // Propagate the numerical solution over the course interval
            y_test = course_method.shifted_euler(y_test, brownian_increment, brownian_area);
        }

        // Reset the numerical solution
        y_test = y0;
    }

    auto finish = std::chrono::high_resolution_clock::now();

    // Obtain the time taken by the speed test
    std::chrono::duration<double> elapsed = finish - start;


    // Display the results in a text file
    ofstream myfile;
    myfile.open ("oscillator_simulation.txt");

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