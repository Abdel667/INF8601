// Abdel Sant'anna. inspired by https://github.com/Wassim27/INF8601/blob/main/TP3/source/heatsim-mpi.c

#include <assert.h>
#include <mpi.h>
#include <stddef.h>

#include "heatsim.h"
#include "log.h"

int heatsim_init(heatsim_t* heatsim, unsigned int dim_x, unsigned int dim_y) {
    /*
     * TODO: Initialiser tous les membres de la structure `heatsim`.
     *       Le communicateur doit être périodique. Le communicateur
     *       cartésien est périodique en X et Y.
     */

    MPI_Comm old_comm = MPI_COMM_WORLD; // Communicateur de base.
    MPI_Comm new_comm; // Communicateur cartésien.
    int ndims = 2; // Nombre de dimensions.
    int dim_size[2] = {dim_x, dim_y}; // Taille de la grille.
    int periods[2] = {1, 1}; // Périodicité (X et Y) activée.
    int reorder = 0; // on autorise pas MPI à changer l'ordre
    int ierr; // Code d'erreur MPI.

    // Création du communicateur cartésien périodique.
    ierr = MPI_Cart_create(old_comm, ndims, dim_size, periods, reorder, &new_comm);
    if (ierr != MPI_SUCCESS) {
        return -1; // Échec de la création du communicateur cartésien.
    }

    // Remplissage de la structure `heatsim_t`.
    heatsim->communicator = new_comm;

    // Obtenir le nombre total de processus.
    ierr = MPI_Comm_size(new_comm, &heatsim->rank_count);
    if (ierr != MPI_SUCCESS) {
        MPI_Comm_free(&new_comm);
        return -1;
    }

    // Obtenir le rang du processus.
    ierr = MPI_Comm_rank(new_comm, &heatsim->rank);
    if (ierr != MPI_SUCCESS) {
        MPI_Comm_free(&new_comm);
        return -1;
    }

    // Obtenir les coordonnées dans le communicateur cartésien.
    ierr = MPI_Cart_coords(new_comm, heatsim->rank, ndims, heatsim->coordinates);
    if (ierr != MPI_SUCCESS) {
        MPI_Comm_free(&new_comm);
        return -1;
    }

    // Identifier les voisins dans les directions cardinales.
    ierr = MPI_Cart_shift(new_comm, 0, 1, &heatsim->rank_west_peer, &heatsim->rank_east_peer);
    if (ierr != MPI_SUCCESS) {
        MPI_Comm_free(&new_comm);
        return -1;
    }

    ierr = MPI_Cart_shift(new_comm, 1, 1, &heatsim->rank_north_peer, &heatsim->rank_south_peer);
    if (ierr != MPI_SUCCESS) {
        MPI_Comm_free(&new_comm);
        return -1;
    }
}

typedef struct data {
  double *data;
} data_t;

MPI_Datatype grid_data_struct(unsigned int size) {
  MPI_Datatype type;

  MPI_Datatype field_types[1] = {MPI_DOUBLE};

  MPI_Aint field_positions[1] = {offsetof(data_t, data)};

  int field_lengths[1] = {size};

  MPI_Type_create_struct(1, field_lengths, field_positions, field_types, &type);
  MPI_Type_commit(&type);

  return type;
}

int heatsim_send_grids(heatsim_t* heatsim, cart2d_t* cart) {
    /*
     * TODO: Envoyer toutes les `grid` aux autres rangs. Cette fonction
     *       est appelé pour le rang 0. Par exemple, si le rang 3 est à la
     *       coordonnée cartésienne (0, 2), alors on y envoit le `grid`
     *       à la position (0, 2) dans `cart`.
     *
     *       Il est recommandé d'envoyer les paramètres `width`, `height`
     *       et `padding` avant les données. De cette manière, le receveur
     *       peut allouer la structure avec `grid_create` directement.
     *
     *       Utilisez `cart2d_get_grid` pour obtenir la `grid` à une coordonnée.
     */

    int ierr;
    MPI_Request request;

    // The main process sends grids to other ranks.
    for (unsigned int i = 1; i < heatsim->num_ranks; i++) {
        int localCoords[2];
        ierr = MPI_Cart_coords(heatsim->communicator, i, 2, localCoords);
        if (ierr != MPI_SUCCESS) {
            fprintf(stderr, "Error getting coordinates for rank %d.\n", i);
            goto fail_exit;
        }

        grid_t* grid = cart2d_get_grid(cart, localCoords[0], localCoords[1]);

        if (!grid) {
            fprintf(stderr, "Error: grid at (%d, %d) is NULL.\n", localCoords[0], localCoords[1]);
            continue;
        }

        unsigned int params[3] = {grid->width, grid->height, (unsigned int)grid->padding};

        // Send grid parameters (width, height, padding).
        MPI_ISend(&params, 3, MPI_UNSIGNED, i, 0, heatsim->communicator, &request);
        ierr = MPI_Wait(&request, MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {
        LOG_ERROR_MPI("Error waiting : ", ret);
        goto fail_exit;
        }
        
        MPI_Datatype data_type = grid_data_struct(grid->width * grid->height);

        // NOTE: Pas de & car grid->data est deja un pointeur
        ierr = MPI_Isend(grid->data, 1, data_type, i, 4, heatsim->communicator, &request);
        if (ret != MPI_SUCCESS) {
        LOG_ERROR_MPI("Error send data : ", ret);
        goto fail_exit;
        }

        ierr = MPI_Wait(&request, MPI_STATUS_IGNORE);
        if (ierr != MPI_SUCCESS) {
        LOG_ERROR_MPI("Error waiting : ", ret);
        goto fail_exit;
        }
        MPI_Type_free(&data_type);
    }

    return 0;

fail_exit:
    return -1;
}

grid_t* heatsim_receive_grid(heatsim_t* heatsim) { //avec MPI_Irecv
    /*
     * TODO: Recevoir un `grid ` du rang 0. Il est important de noté que
     *       toutes les `grid` ne sont pas nécessairement de la même
     *       dimension (habituellement ±1 en largeur et hauteur). Utilisez
     *       la fonction `grid_create` pour allouer un `grid`.
     *
     *       Utilisez `grid_create` pour allouer le `grid` à retourner.
     */
    // Postez les réceptions asynchrones
    int ierr = MPI_SUCCESS;
    unsigned int params[3];
    MPI_Request request;

    ierr = MPI_Irecv(&dimensions, 3, MPI_UNSIGNED, 0, 0, heatsim->communicator, &request);

    ierr = MPI_Wait(&request, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {
        LOG_ERROR_MPI("Error Wait receive data : ", ret);
        goto fail_exit;
    }
    grid_t *newGrid = grid_create(params[0], params[1], params[2]);

    // partie 2

    MPI_Datatype data_type = grid_data_struct(newGrid->width_padded * newGrid->height_padded);

    ierr = MPI_Irecv(newGrid->data, newGrid->width_padded * newGrid->height_padded,
                    data_type, 0, 4, heatsim->communicator, &request);
    if (ierr != MPI_SUCCESS) {
        LOG_ERROR_MPI("Error receive data : ", ret);
        goto fail_exit;
    }

    ierr = MPI_Wait(&request, MPI_STATUS_IGNORE);
    if (ierr != MPI_SUCCESS) {
        LOG_ERROR_MPI("Error Wait receive data : ", ret);
        goto fail_exit;
    }

    return newGrid;

fail_exit:
    return NULL;
}

int heatsim_exchange_borders(heatsim_t* heatsim, grid_t* grid) {
    assert(grid->padding == 1);

    /*
     * TODO: Échange les bordures de `grid`, excluant le rembourrage, dans le
     *       rembourrage du voisin de ce rang. Par exemple, soit la `grid`
     *       4x4 suivante,
     *
     *                            +-------------+
     *                            | x x x x x x |
     *                            | x A B C D x |
     *                            | x E F G H x |
     *                            | x I J K L x |
     *                            | x M N O P x |
     *                            | x x x x x x |
     *                            +-------------+
     *
     *       où `x` est le rembourrage (padding = 1). Ce rang devrait envoyer
     *
     *        - la bordure [A B C D] au rang nord,
     *        - la bordure [M N O P] au rang sud,
     *        - la bordure [A E I M] au rang ouest et
     *        - la bordure [D H L P] au rang est.
     *
     *       Ce rang devrait aussi recevoir dans son rembourrage
     *
     *        - la bordure [A B C D] du rang sud,
     *        - la bordure [M N O P] du rang nord,
     *        - la bordure [A E I M] du rang est et
     *        - la bordure [D H L P] du rang ouest.
     *
     *       Après l'échange, le `grid` devrait avoir ces données dans son
     *       rembourrage provenant des voisins:
     *
     *                            +-------------+
     *                            | x m n o p x |
     *                            | d A B C D a |
     *                            | h E F G H e |
     *                            | l I J K L i |
     *                            | p M N O P m |
     *                            | x a b c d x |
     *                            +-------------+
     *
     *       Utilisez `grid_get_cell` pour obtenir un pointeur vers une cellule.
     */

fail_exit:
    return -1;
}

int heatsim_send_result(heatsim_t* heatsim, grid_t* grid) {
    assert(grid->padding == 0);

    /*
     * TODO: Envoyer les données (`data`) du `grid` résultant au rang 0. Le
     *       `grid` n'a aucun rembourage (padding = 0);
     */

fail_exit:
    return -1;
}

int heatsim_receive_results(heatsim_t* heatsim, cart2d_t* cart) {
    /*
     * TODO: Recevoir toutes les `grid` des autres rangs. Aucune `grid`
     *       n'a de rembourage (padding = 0).
     *
     *       Utilisez `cart2d_get_grid` pour obtenir la `grid` à une coordonnée
     *       qui va recevoir le contenue (`data`) d'un autre noeud.
     */

fail_exit:
    return -1;
}
