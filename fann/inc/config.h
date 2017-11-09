//#include "thyroid_test.h"
#include "fann.h"

#define NUM_LAYERS 3
#define NUM_NEURONS                       32 //XXX static
#define TOT_CONNECTIONS 128

#define cascade_activation_functions_count_COUNT 10
#define cascade_activation_steepnesses_count_COUNT 4
extern struct fann ann_mem;
extern fann_type neurons[][3];
extern fann_type connections[][2];
extern fann_type ann_out[NUM_NEURONS];
//extern enum fann_activationfunc_enum fann_activationfunc_enum_arr[cascade_activation_functions_count_COUNT];
//extern fann_type cascade_activation_steepnesses_arr[cascade_activation_steepnesses_count_COUNT];
//extern struct fann_layer_arr[NUM_LAYERS];

/* Name of package */
/* #undef PACKAGE */

/* Version number of package */
/* #undef VERSION */

/* Define for the x86_64 CPU famyly */
/* #undef X86_64 */
