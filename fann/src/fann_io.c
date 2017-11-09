/*
 *******************************************************************************
 * fann_io.c
 *
 * FANN training data reading function ported for embedded devices.
 *
 * Created on: Oct 23, 2017
 *    Authors: Dimitris Patoukas, Carlo Delle Donne
 *******************************************************************************
 */

#include <msp430.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <stdint.h>
#include "config.h"
#include "fann.h"
#include "fann_data.h"
#include "thyroid_trained.h"

//#pragma PERSISTENT(cascade_activation_functions)
//#pragma PERSISTENT(cascade_activation_steepnesses)
#pragma PERSISTENT(weights)
#pragma PERSISTENT(ann_out)
#pragma PERSISTENT(ann_neurons)
#pragma PERSISTENT(ann_layers)
#pragma PERSISTENT(ann_conn)

fann_type ann_out[NUM_NEURONS];

const enum fann_activationfunc_enum cascade_activation_functions[CASCADE_ACTIVATION_FUNCTIONS_COUNT]= {
    CASCADE_ACTIVATION_FUNCTION_1,
    CASCADE_ACTIVATION_FUNCTION_2,
    CASCADE_ACTIVATION_FUNCTION_3,
    CASCADE_ACTIVATION_FUNCTION_4,
    CASCADE_ACTIVATION_FUNCTION_5,
    CASCADE_ACTIVATION_FUNCTION_6,
    CASCADE_ACTIVATION_FUNCTION_7,
    CASCADE_ACTIVATION_FUNCTION_8,
    CASCADE_ACTIVATION_FUNCTION_9,
    CASCADE_ACTIVATION_FUNCTION_10
};

const fann_type cascade_activation_steepnesses[CASCADE_ACTIVATION_STEEPNESSES_COUNT] = {
    CASCADE_ACTIVATION_STEEPNESS_1,
    CASCADE_ACTIVATION_STEEPNESS_2,
    CASCADE_ACTIVATION_STEEPNESS_3,
    CASCADE_ACTIVATION_STEEPNESS_4
};

fann_type weights[TOT_CONNECTIONS];

struct fann_neuron ann_neurons[NUM_NEURONS];
struct fann_layer ann_layers[NUM_LAYERS];
struct fann_neuron *ann_conn[TOT_CONNECTIONS];
/**
 * INTERNAL FUNCTION
 *
 * The ANN is created from constant values contained in 
 * database/<example>_trained.h
 * where <example> is the subject example (e.g. xor, thyroid, etc.)
 */
struct fann *fann_create_msp430()
{



    /* Allocate network. */
    struct fann *ann = &ann_mem;
    uint8_t input_neuron;
    uint8_t i;


    struct fann_neuron *first_neuron, *neuron_it, *last_neuron, **connected_neurons;

    /* Assign parameters. */

    ann->learning_rate = LEARNING_RATE;
    ann->connection_rate = CONNECTION_RATE;

    ann->network_type = (enum fann_nettype_enum) NETWORK_TYPE;
    ann->learning_momentum = LEARNING_MOMENTUM;
    ann->training_algorithm = (enum fann_train_enum) TRAINING_ALGORITHM;
    ann->train_error_function = (enum fann_errorfunc_enum) TRAIN_ERROR_FUNCTION;
    ann->train_stop_function = (enum fann_stopfunc_enum) TRAIN_STOP_FUNCTION;

    ann->cascade_output_change_fraction = CASCADE_OUTPUT_CHANGE_FRACTION;
    ann->quickprop_decay = QUICKPROP_DECAY;
    ann->quickprop_mu = QUICKPROP_MU;
    ann->rprop_increase_factor = RPROP_INCREASE_FACTOR;
    ann->rprop_decrease_factor = RPROP_DECREASE_FACTOR;
    ann->rprop_delta_min = RPROP_DELTA_MIN;
    ann->rprop_delta_max = RPROP_DELTA_MAX;
    ann->rprop_delta_zero = RPROP_DELTA_ZERO;
    ann->cascade_output_stagnation_epochs = CASCADE_OUTPUT_STAGNATION_EPOCHS;
    ann->cascade_candidate_change_fraction = CASCADE_CANDIDATE_CHANGE_FRACTION;
    ann->cascade_candidate_stagnation_epochs = CASCADE_CANDIDATE_STAGNATION_EPOCHS;
    ann->cascade_max_out_epochs = CASCADE_MAX_OUT_EPOCHS;
    ann->cascade_min_out_epochs = CASCADE_MIN_OUT_EPOCHS;
    ann->cascade_max_cand_epochs = CASCADE_MAX_CAND_EPOCHS;
    ann->cascade_min_cand_epochs = CASCADE_MIN_CAND_EPOCHS;
    ann->cascade_num_candidate_groups = CASCADE_NUM_CANDIDATE_GROUPS;
    ann->bit_fail_limit = BIT_FAIL_LIMIT;
    ann->cascade_candidate_limit = CASCADE_CANDIDATE_LIMIT;
    ann->cascade_weight_multiplier = CASCADE_WEIGHT_MULTIPLIER;

    ann->cascade_activation_functions_count = CASCADE_ACTIVATION_FUNCTIONS_COUNT;
    // WARNING: dynamic allocation!
    ann->cascade_activation_functions = cascade_activation_functions;
    if (ann->cascade_activation_functions == NULL) {
        // fann_error((struct fann_error*)ann, FANN_E_CANT_ALLOCATE_MEM);
        //fann_destroy(ann);
        return NULL;
    }

#ifdef DEBUG_MALLOC
    //printf("Re-allocated %u bytes for activation functions.\n",
    //        ann->cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum));
#endif // DEBUG_MALLOC


    ann->cascade_activation_steepnesses_count = CASCADE_ACTIVATION_STEEPNESSES_COUNT;
    // WARNING: for
    ann->cascade_activation_steepnesses = cascade_activation_steepnesses;

#ifdef FIXEDFANN
    fann_update_stepwise(ann);
#endif // FIXEDFANN

#ifdef DEBUG
    printf("Creating network with %d layers\n", NUM_LAYERS);
    printf("Input\n");
#endif // DEBUG

    ann->first_layer = &ann_layers[0];
    ann->last_layer = &ann_layers[NUM_LAYERS-1];

    ann->first_layer->first_neuron=(void *)&ann_neurons[0];
    ann->first_layer->last_neuron=(void *)&ann_neurons[LAYER_SIZE_1-1];
    ann->last_layer->first_neuron=(void *)&ann_neurons[NUM_NEURONS-LAYER_SIZE_3-1];
    ann->last_layer->last_neuron=(void *)&ann_neurons[NUM_NEURONS-1];

    ann->total_neurons = NUM_NEURONS;



    ann->num_input = (unsigned int) (ann->first_layer->last_neuron - ann->first_layer->first_neuron - 1);
    ann->num_output = (unsigned int) ((ann->last_layer - 1)->last_neuron - (ann->last_layer - 1)->first_neuron);
    ann->output = ann_out;
    ann->total_neurons_allocated = ann->total_neurons;

    if (ann->network_type == FANN_NETTYPE_LAYER) {
        // One too many (bias) in the output layer
        ann->num_output--;
    }

    // TODO: redundant on example case, fix for portability
/*
#ifndef FIXEDFANN
#define SCALE_LOAD( what, where ) \
    fann_skip( #what "_" #where "=" ); \
    for (i = 0; i < ann->num_##where##put; i++) { \
        if (fscanf( conf, "%f ", (float *)&ann->what##_##where[ i ] ) != 1) { \
            fann_error((struct fann_error *) ann, FANN_E_CANT_READ_CONFIG, #what "_" #where, configuration_file); \
            fann_destroy(ann); \
            return NULL; \
        } \
    }

    if (SCALE_INCLUDED == 1) {
       fann_allocate_scale(ann);
       SCALE_LOAD( scale_mean,         in )
       SCALE_LOAD( scale_deviation,    in )
       SCALE_LOAD( scale_new_min,      in )
       SCALE_LOAD( scale_factor,       in )

       SCALE_LOAD( scale_mean,         out )
       SCALE_LOAD( scale_deviation,    out )
       SCALE_LOAD( scale_new_min,      out )
       SCALE_LOAD( scale_factor,       out )
   }
#undef SCALE_LOAD
#endif
*/

    // WARNING: dynamic allocation!
    //fann_allocate_neurons(ann);


    last_neuron = (ann->last_layer - 1)->last_neuron;
    for (neuron_it = ann->first_layer->first_neuron; neuron_it != last_neuron; neuron_it++) {
        //num_connections = neurons[i][0];
        neuron_it->activation_steepness = neurons[i][2];
        neuron_it->activation_function = (enum fann_activationfunc_enum) neurons[i][1];//tmp_val;
        neuron_it->first_con = ann->total_connections;
        ann->total_connections += neurons[i][0];//num_connections;
        neuron_it->last_con = ann->total_connections;
        i++;
    }

    ann->weights = weights;
    ann->total_connections = TOT_CONNECTIONS;
    ann->connections = ann_conn;
    ann->total_connections_allocated = ann->total_connections;

    connected_neurons = ann->connections;
    first_neuron = ann->first_layer->first_neuron;

    for (i = 0; i < ann->total_connections; i++) {
        input_neuron = connections[i][0];
        ann->weights[i] = connections[i][1];
        connected_neurons[i] = first_neuron + input_neuron;
    }

    return ann;
}


/**
 * Create network from header file.
 */
FANN_EXTERNAL struct fann *FANN_API fann_create_from_header()
{
    struct fann *ann;

    ann = fann_create_msp430();

    return ann;
}
