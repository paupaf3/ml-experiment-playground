#include <iostream>
#include <cmath>

int main() {
	double pred, target, error;
	// 'std::out' is the standard character output
	// '<<' is the insertion operator (think of it as "sending data to the screen")
	std::cout << "--- Interactive Loss Calculator ---" << std::endl;
	
	std::cout << "Enter prediciton: ";
	// 'std::cin' is standard input
	// '>>' is the extraction operator (taking data from)
	std::cin >> pred;

	std::cout << "Enter target: ";
	std::cin >> target;

	// Calculate squared error
	// (pred - target)^2
	error = std::pow((pred - target), 2);
	std::cout << "Squared error: " << error << std::endl;

	return 0;
	
}
