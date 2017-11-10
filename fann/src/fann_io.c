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

fann_type ann_out[NUM_NEURONS]={0};

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

fann_type weights[TOT_CONNECTIONS]={0};

struct fann_neuron ann_neurons[NUM_NEURONS]={0};
struct fann_layer ann_layers[NUM_LAYERS]={0};
struct fann_neuron *ann_conn[TOT_CONNECTIONS]={0};
/**
 * INTERNAL FUNCTION
 *
 * The ANN is created from constant values contained in 
 * database/<example>_trained.h
 * where <example> is the subject example (e.g. xor, thyroid, etc.)
 */
void fann_create_msp430()
{



    /* Allocate network. */
    //struct fann *ann = &ann_mem;
    uint8_t input_neuron;
    #pragma PERSISTENT(i)
    uint8_t i;
#pragma PERSISTENT(num_connections)
#pragma PERSISTENT(tmp_val)
    uint8_t num_connections;
    uint8_t tmp_val;

    struct fann_neuron *first_neuron, *neuron_it, *last_neuron, **connected_neurons;

    /* Assign parameters. */

    ann_mem.learning_rate = LEARNING_RATE;
    ann_mem.connection_rate = CONNECTION_RATE;

    ann_mem.network_type = (enum fann_nettype_enum) NETWORK_TYPE;
    ann_mem.learning_momentum = LEARNING_MOMENTUM;
    ann_mem.training_algorithm = (enum fann_train_enum) TRAINING_ALGORITHM;
    ann_mem.train_error_function = (enum fann_errorfunc_enum) TRAIN_ERROR_FUNCTION;
    ann_mem.train_stop_function = (enum fann_stopfunc_enum) TRAIN_STOP_FUNCTION;

    ann_mem.cascade_output_change_fraction = CASCADE_OUTPUT_CHANGE_FRACTION;
    ann_mem.quickprop_decay = QUICKPROP_DECAY;
    ann_mem.quickprop_mu = QUICKPROP_MU;
    ann_mem.rprop_increase_factor = RPROP_INCREASE_FACTOR;
    ann_mem.rprop_decrease_factor = RPROP_DECREASE_FACTOR;
    ann_mem.rprop_delta_min = RPROP_DELTA_MIN;
    ann_mem.rprop_delta_max = RPROP_DELTA_MAX;
    ann_mem.rprop_delta_zero = RPROP_DELTA_ZERO;
    ann_mem.cascade_output_stagnation_epochs = CASCADE_OUTPUT_STAGNATION_EPOCHS;
    ann_mem.cascade_candidate_change_fraction = CASCADE_CANDIDATE_CHANGE_FRACTION;
    ann_mem.cascade_candidate_stagnation_epochs = CASCADE_CANDIDATE_STAGNATION_EPOCHS;
    ann_mem.cascade_max_out_epochs = CASCADE_MAX_OUT_EPOCHS;
    ann_mem.cascade_min_out_epochs = CASCADE_MIN_OUT_EPOCHS;
    ann_mem.cascade_max_cand_epochs = CASCADE_MAX_CAND_EPOCHS;
    ann_mem.cascade_min_cand_epochs = CASCADE_MIN_CAND_EPOCHS;
    ann_mem.cascade_num_candidate_groups = CASCADE_NUM_CANDIDATE_GROUPS;
    ann_mem.bit_fail_limit = BIT_FAIL_LIMIT;
    ann_mem.cascade_candidate_limit = CASCADE_CANDIDATE_LIMIT;
    ann_mem.cascade_weight_multiplier = CASCADE_WEIGHT_MULTIPLIER;

    ann_mem.cascade_activation_functions_count = CASCADE_ACTIVATION_FUNCTIONS_COUNT;
    // WARNING: dynamic allocation!
    ann_mem.cascade_activation_functions = cascade_activation_functions;


#ifdef DEBUG_MALLOC
    //printf("Re-allocated %u bytes for activation functions.\n",
    //        ann_mem.cascade_activation_functions_count * sizeof(enum fann_activationfunc_enum));
#endif // DEBUG_MALLOC


    ann_mem.cascade_activation_steepnesses_count = CASCADE_ACTIVATION_STEEPNESSES_COUNT;
    // WARNING: for
    ann_mem.cascade_activation_steepnesses = cascade_activation_steepnesses;

#ifdef FIXEDFANN
    fann_update_stepwise(ann);
#endif // FIXEDFANN

#ifdef DEBUG
    printf("Creating network with %d layers\n", NUM_LAYERS);
    printf("Input\n");
#endif // DEBUG

    ann_mem.first_layer = &ann_layers[0];
    ann_mem.last_layer = &ann_layers[NUM_LAYERS-1];

    ann_mem.first_layer->first_neuron=(void *)&ann_neurons[0];
    ann_mem.first_layer->last_neuron=(void *)&ann_neurons[LAYER_SIZE_1-1];
    ann_mem.last_layer->first_neuron=(void *)&ann_neurons[NUM_NEURONS-LAYER_SIZE_3-1];
    ann_mem.last_layer->last_neuron=(void *)&ann_neurons[NUM_NEURONS-1];

    (ann_mem.last_layer-1)->first_neuron=(void *)&ann_neurons[LAYER_SIZE_1];
    (ann_mem.last_layer-1)->last_neuron=(void *)&ann_neurons[LAYER_SIZE_1+LAYER_SIZE_2-1];

    ann_mem.total_neurons = NUM_NEURONS;

    last_neuron = (ann_mem.last_layer)->last_neuron;
    for (neuron_it = ann_mem.first_layer->first_neuron; neuron_it != last_neuron; neuron_it++) {
        num_connections = neurons[i][0];
        neuron_it->activation_steepness = neurons[i][2];
        tmp_val = (enum fann_activationfunc_enum) neurons[i][1];
        i++;
        neuron_it->activation_function = tmp_val;
        neuron_it->first_con = ann_mem.total_connections;
        ann_mem.total_connections += num_connections;
        neuron_it->last_con = ann_mem.total_connections;

    }

    ann_mem.num_input = (unsigned int) (ann_mem.first_layer->last_neuron - ann_mem.first_layer->first_neuron - 1);
    ann_mem.num_output = (unsigned int) ((ann_mem.last_layer - 1)->last_neuron - (ann_mem.last_layer - 1)->first_neuron);
    ann_mem.output = ann_out;
    ann_mem.total_neurons_allocated = ann_mem.total_neurons;

    if (ann_mem.network_type == FANN_NETTYPE_LAYER) {
        // One too many (bias) in the output layer
        ann_mem.num_output--;
    }

    // TODO: redundant on example case, fix for portability
/*
#ifndef FIXEDFANN
#define SCALE_LOAD( what, where ) \
    fann_skip( #what "_" #where "=" ); \
    for (i = 0; i < ann_mem.num_##where##put; i++) { \
        if (fscanf( conf, "%f ", (float *)&ann_mem.what##_##where[ i ] ) != 1) { \
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

    ann_mem.weights = weights;
    ann_mem.total_connections = TOT_CONNECTIONS;
    ann_mem.connections = ann_conn;
    ann_mem.total_connections_allocated = ann_mem.total_connections;

    connected_neurons = ann_mem.connections;
    first_neuron = ann_mem.first_layer->first_neuron;

    for (i = 0; i < ann_mem.total_connections; i++) {
        input_neuron = connections[i][0];
        ann_mem.weights[i] = connections[i][1];
        connected_neurons[i] = first_neuron + input_neuron;
    }

}


/**
 * Create network from header file.
 */
FANN_EXTERNAL void FANN_API fann_create_from_header()
{
    fann_create_msp430();
}
