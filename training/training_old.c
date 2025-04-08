/*
 *  training.c
 *
 *  Arxiu reutilitzat de l'assignatura de Computació d'Altes Prestacions de l'Escola d'Enginyeria de la Universitat Autònoma de Barcelona
 *  Created on: 31 gen. 2019
 *  Last modified: fall 24 (curs 24-25)
 *  Author: ecesar, asikora
 *  Modified: Blanca Llauradó, Christian Germer
 *
 *  Descripció:
 *  Funcions per entrenar la xarxa neuronal.
 *
 */

 #include "training.h"

 #include <math.h>
 
 // General Notes
 
 // 03/04/2025
 // Started properly using pragmas on every function I could.
 // Unfortunately I forgot for some reason that I need to use parallel to actually parallelise
 // Brought time down to 9.9s but due to lack of comments I'll have to revisit the code.
 // Probably issues lie within the forward_prop().
 
 // 05/04/2025
 // First issue: use parallel to properly use more CPU's
 // After some of that the current time is around 10.3
 /**
  * @brief Iniciatlitza la capa incial de la xarxa (input layer) amb l'entrada
  * que volem reconeixer.
  *
  * @param i Índex de l'element del conjunt d'entrenament que farem servir.
  **/
 void feed_input(int i) {
     // 05/04/2025 Daniel Gil
     // I'm gonna start revisiting the whole code
     // 05/04/2024 adding parallel affects not very much on 2 threads
     # pragma parallel omp for
     for (int j = 0; j < num_neurons[0]; j++)
         lay[0].actv[j] = input[i][j];
 }
 
 /**
  * @brief Propagació dels valors de les neurones de l'entrada (valors a la input
  * layer) a la resta de capes de la xarxa fins a obtenir una predicció (sortida)
  *
  * @details La capa d'entrada (input layer = capa 0) ja ha estat inicialitzada
  * amb els valors de l'entrada que volem reconeixer. Així, el for més extern
  * (sobre i) recorre totes les capes de la xarxa a partir de la primera capa
  * hidden (capa 1). El for intern (sobre j) recorre les neurones de la capa i
  * calculant el seu valor d'activació [lay[i].actv[j]]. El valor d'activació de
  * cada neurona depén de l'exitació de la neurona calculada en el for més intern
  * (sobre k) [lay[i].z[j]]. El valor d'exitació s'inicialitza amb el biax de la
  * neurona corresponent [j] (lay[i].bias[j]) i es calcula multiplicant el valor
  * d'activació de les neurones de la capa anterior (i-1) pels pesos de
  * les connexions (out_weights) entre les dues capes. Finalment, el valor
  * d'activació de la neurona (j) es calcula fent servir la funció RELU
  * (REctified Linear Unit) si la capa (j) és una capa oculta (hidden) o la
  * funció Sigmoid si es tracte de la capa de sortida.
  *
  */
 
 void forward_prop() {
     //# pragma omp for // lets try this shit first this adds like 1 sec
     int i, j, k = 0;
     // # pragma omp parallel for reduction(+:s)
     //# pragma omp parallel for
     for (int i = 1; i < num_layers; i++) {
 
         // This actually makes it slower
         # pragma omp parallel for
         for (int j = 0; j < num_neurons[i]; j++) {
             lay[i].z[j] = lay[i].bias[j];
 
             // 04/05/2024 Testing if this makes sense or makes it go faster, it does not
             // # pragma omp for
             //
             // 05/05/2025
             // Im gonna try modify some code here to use reduction
             for (int k = 0; k < num_neurons[i - 1]; k++)
                 lay[i].z[j] += ((lay[i - 1].out_weights[j * num_neurons[i - 1] + k]) * (lay[i - 1].actv[k]));
             if (i <
                 num_layers - 1)  // Relu Activation Function for Hidden Layers
                 lay[i].actv[j] = ((lay[i].z[j]) < 0) ? 0 : lay[i].z[j];
             else  // Sigmoid Activation Function for Output Layer
                 lay[i].actv[j] = 1 / (1 + exp(-lay[i].z[j]));
         }
     }
 }
 
 /**
  * @brief Calcula el gradient que es necessari aplicar als pesos de les
  * connexions entre neurones per corregir els errors de predicció
  *
  * @details Calcula dos vectors de correcció per cada capa de la xarxa, un per
  * corregir els pesos de les connexions de la neurona (j) amb la capa anterior
  *          (lay[i-1].dw[j]) i un segon per corregir el biax de cada neurona de
  * la capa actual (lay[i].bias[j]). Hi ha un tractament diferent per la capa de
  * sortida (num_layesr -1) perquè aquest és l'única cas en el que l'error es
  * conegut (lay[num_layers-1].actv[j] - desired_outputs[p][j]). Això es pot
  * veure en els dos primers fors. Per totes les capes ocultes (hidden layers) no
  * es pot saber el valor d'activació esperat per a cada neurona i per tant es fa
  * una estimació. Aquest càlcul es fa en el doble for que recorre totes les
  * capes ocultes (sobre i) neurona a neurona (sobre j). Es pot veure com en cada
  * cas es fa una estimació de quines hauríen de ser les activacions de les
  * neurones de la capa anterior (lay[i-1].dactv[k] = lay[i-1].out_weights[j*
  * num_neurons[i-1] + k] * lay[i].dz[j];), excepte pel cas de la capa d'entrada
  * (input layer) que és coneguda (imatge d'entrada).
  *
  */
 void back_prop(int p) {
     // Output Layer
     // Also does not seem to affect much the overall time
     // It does something but its not much. Starting using CPUs
     // 04/05/2025
     // This function seems as optimized as it could be
     // 06/05/2025
     // Yeah this pragma is alright here. It sabes 0.1 seconds but is worth it.
     # pragma omp parallel for
     for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
         lay[num_layers - 1].dz[j] =
             (lay[num_layers - 1].actv[j] - desired_outputs[p][j]) *
             (lay[num_layers - 1].actv[j]) * (1 - lay[num_layers - 1].actv[j]);
         lay[num_layers - 1].dbias[j] = lay[num_layers - 1].dz[j];
     }
     # pragma omp parallel for collapse(2)
     for (int j = 0; j < num_neurons[num_layers - 1]; j++) {
         for (int k = 0; k < num_neurons[num_layers - 2]; k++) {
             lay[num_layers - 2].dw[j * num_neurons[num_layers - 2] + k] =
                 (lay[num_layers - 1].dz[j] * lay[num_layers - 2].actv[k]);
             lay[num_layers - 2].dactv[k] =
                 lay[num_layers - 2]
                     .out_weights[j * num_neurons[num_layers - 2] + k] *
                 lay[num_layers - 1].dz[j];
         }
     }
 
     // Hidden Layers
     int i, j, k;
     // 04/05/2025 Still loking for a way to parallelize this with collapse
     // This gives an error IDK how to solve it
     # pragma omp parallel for
     for (int i = num_layers - 2; i > 0; i--) {
         for (int j = 0; j < num_neurons[i]; j++) {
             lay[i].dz[j] = (lay[i].z[j] >= 0) ? lay[i].dactv[j] : 0;
 
             for (int k = 0; k < num_neurons[i - 1]; k++) {
                 lay[i - 1].dw[j * num_neurons[i - 1] + k] =
                     lay[i].dz[j] * lay[i - 1].actv[k];
 
                 if (i > 1)
                     lay[i - 1].dactv[k] =
                         lay[i - 1].out_weights[j * num_neurons[i - 1] + k] *
                         lay[i].dz[j];
             }
             lay[i].dbias[j] = lay[i].dz[j];
         }
     }
 }
 
 /**
  * @brief Actualitza els vectors de pesos (out_weights) i de biax (bias) de cada
  * etapa d'acord amb els càlculs fet a la funció de back_prop i el factor
  * d'aprenentatge alpha
  *
  * @see back_prop
  */
 void update_weights(void) { // no afedta mucho separar el otro for
 
     for (int i = 0; i < num_layers - 1; i++) {
         # pragma omp parallel for collapse(2)
         for (int j = 0; j < num_neurons[i + 1]; j++)
             for (int k = 0; k < num_neurons[i]; k++)  // Update Weights
                 lay[i].out_weights[j * num_neurons[i] + k] =
                     (lay[i].out_weights[j * num_neurons[i] + k]) -
                     (alpha * lay[i].dw[j * num_neurons[i] + k]);
     }
     # pragma omp parallel for
     for (int i = 0; i < num_layers - 1; i++)
         for (int j = 0; j < num_neurons[i]; j++)  // Update Bias
             lay[i].bias[j] = lay[i].bias[j] - (alpha * lay[i].dbias[j]);
 
 }
 