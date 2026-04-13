#include <stdio.h>
#include <math.h>

int main () {
	double pred, target, error;

	printf("--- Interactive Loss Calculator ---\n");

	// We use %lf because a 'double' is a 'long float'
	printf("Enter prediciton: ");
	scanf("%lf", &pred);

	printf("Enter target: ");
	scanf("%lf", &target);

	// Calculate squared error
	// (pred - target)^2
	error = pow((pred - target), 2);
	printf("Squared error: %f\n", error);

	return 0;
}
