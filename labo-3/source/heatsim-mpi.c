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

    // Le processus principal envoie les grilles aux autres rangs.
    for (unsigned int x = 0; x < cart->grid_x_count; x++) {
        for (unsigned int y = 0; y < cart->grid_y_count; y++) {
            // Obtenez la grille à la position (x, y).
            grid_t* grid = cart2d_get_grid(cart, x, y);
            if (!grid) {
                fprintf(stderr, "Erreur : la grille à (%d, %d) est NULL.\n", x, y);
                continue;
            }

            // Déterminez le rang du processus correspondant aux coordonnées (x, y).
            int coords[2] = {x, y};
            int dest_rank;
            ierr = MPI_Cart_rank(heatsim->communicator, coords, &dest_rank);
            if (ierr != MPI_SUCCESS) {
                fprintf(stderr, "Erreur : impossible d'obtenir le rang pour (%d, %d).\n", x, y);
                continue;
            }

            // Si c'est le rang 0, pas besoin d'envoyer (car il s'agit du processus principal lui-même).
            if (dest_rank == 0) {
                continue;
            }

            // Préparez les paramètres à envoyer.
            unsigned int params[3] = {grid->width, grid->height, (unsigned int)grid->padding};

            // Envoyez les paramètres de la grille (width, height, padding).
            MPI_Send(params, 3, MPI_UNSIGNED, dest_rank, 0, heatsim->communicator);

            // Envoyez les données de la grille.
            int data_size = grid->width * grid->height;
            MPI_Send(grid->data, data_size, MPI_DOUBLE, dest_rank, 0, heatsim->communicator);
        }
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
    int* recv_buf = malloc((size - 1) * sizeof(int));
    MPI_Irecv(&recv_buf[i - 1], 1, MPI_INT, i, tag, MPI_COMM_WORLD, &requests[i - 1]);
    free(recv_buf);

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
