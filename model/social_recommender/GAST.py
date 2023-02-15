import numpy as np
import scipy.sparse as sp
from model.AbstractRecommender import SocialAbstractRecommender
from util import DataIterator, timer
from util.tool import csr_to_user_dict_bytime
import tensorflow as tf
from util.cython.random_choice import batch_randint_choice
from util import pad_sequences
epsilon = 1e-9


def l2_distance(a, b, name="euclidean_distance"):
    return tf.norm(a - b, ord='euclidean', axis=-1, name=name)
class GAST(SocialAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(GAST, self).__init__(dataset, conf)
        self.dataset = dataset
        self.num_users, self.num_items = dataset.train_matrix.shape

        self.lr = conf["lr"]
        self.l2_reg = conf["l2_reg"]
        self.l2_W = conf["l2_w"]
        self.embedding_size = conf["embedding_size"]
        self.seq_L = conf["seq_l"]
        self.batch_size = conf["batch_size"]
        self.epochs = conf["epochs"]
        self.SocialtrainDict = self._get_SocialDict()
        # GCN's hyperparameters
        self.n_layers = conf['n_layers']
        self.norm_adj = self.create_adj_mat()

        self.sess = sess

    def _get_SocialDict(self):
        # find items rated by trusted neighbors only
        SocialDict = {}
        for u in range(self.num_users):
            trustors = self.social_matrix[u].indices
            if len(trustors) > 0:
                SocialDict[u] = trustors.tolist()
            else:
                SocialDict[u] = []
        return SocialDict

    @timer
    def create_adj_mat(self):
        user_list, item_list = self.dataset.get_train_interactions()
        user_np = np.array(user_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        print('use the pre adjcency matrix')

        return adj_matrix
    def _create_gcn_embed(self):
        adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)
        ego_embeddings = tf.concat([self.embeddings["user_embeddings"], self.embeddings["item_embeddings"]], axis=0)
        initial_user_embeddings = tf.tile(tf.expand_dims(self.embeddings["user_embeddings"], 1),
                                          tf.stack([1, self.n_layers, 1]))  # b*K*d
        initial_item_embeddings = tf.tile(tf.expand_dims(self.embeddings["item_embeddings"], 1),
                                          tf.stack([1, self.n_layers, 1]))  # b*K*d
        all_embeddings = []
        for k in range(0, self.n_layers):
            side_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings, name="sparse_dense")
            # transformed sum messages of neighbors.
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items], 0)

        user_output = tf.matmul(tf.reshape(initial_user_embeddings, [-1, self.embedding_size]), self.weights["GCN_u_w1"]) + \
                      tf.matmul(tf.reshape(u_g_embeddings, [-1, self.embedding_size]), self.weights["GCN_u_w2"])

        user_output = tf.reshape(user_output, [-1, self.n_layers, self.embedding_size])
        user_output = tf.nn.sigmoid(user_output)
        user_final_embeddings = tf.reduce_sum(
            tf.multiply(u_g_embeddings, user_output), axis=1)

        item_output = tf.matmul(tf.reshape(initial_item_embeddings, [-1, self.embedding_size]), self.weights["GCN_i_w1"]) + \
                      tf.matmul(tf.reshape(i_g_embeddings, [-1, self.embedding_size]), self.weights["GCN_i_w2"])

        item_output = tf.reshape(item_output, [-1, self.n_layers, self.embedding_size])
        item_output = tf.nn.sigmoid(item_output)
        item_final_embeddings = tf.reduce_sum(
            tf.multiply(i_g_embeddings, item_output), axis=1)
        return user_final_embeddings, item_final_embeddings

    def _normalize_spmat(self, adj_mat):
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        print('use the pre adjcency matrix')
        return adj_matrix

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_placeholder(self):
        self.user_ph = tf.placeholder(tf.int32, [None, ], name="user")
        self.item_seq_ph = tf.placeholder(tf.int32, [None, self.seq_L], name="item_seq")
        self.social_user_input = tf.placeholder(tf.int32, [None, None], name="social_user_input")
        self.item_pos_ph = tf.placeholder(tf.int32, [None, ], name="item_pos")
        self.item_neg_ph = tf.placeholder(tf.int32, [None, ], name="item_neg")

    def _create_variable(self):
        self.weights = dict()
        self.embeddings = dict()

        embeding_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        user_embeddings = tf.Variable(embeding_initializer([self.num_users, self.embedding_size]), dtype=tf.float32)
        self.embeddings.setdefault("user_embeddings", user_embeddings)

        self.c1 = tf.Variable(embeding_initializer(shape=[self.num_users, self.embedding_size]),
                              dtype=tf.float32)  # (users, embedding_size)
        self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
        self.social_embeddings = tf.concat([self.c1, self.c2], 0, name='social_embeddings')

        item_embeddings = tf.Variable(embeding_initializer([self.num_items, self.embedding_size]), dtype=tf.float32)
        self.embeddings.setdefault("item_embeddings", item_embeddings)

        seq_embeddings = tf.Variable(embeding_initializer([self.num_items, self.embedding_size]), dtype=tf.float32)
        self.embeddings.setdefault("seq_item_embeddings", seq_embeddings)
        self.d2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='d2')
        self.seq_item_embeddings = tf.concat([seq_embeddings, self.d2], 0)

        self.global_embedding = tf.Variable(tf.truncated_normal(shape=[1, self.embedding_size], mean=0.0, stddev=0.01),
                                            name='global_embedding', dtype=tf.float32)
        # position embedding
        position_embeddings = tf.Variable(embeding_initializer([self.seq_L, self.embedding_size]), dtype=tf.float32)
        self.embeddings.setdefault("position_embeddings", position_embeddings)

        self.item_biases = tf.Variable(tf.truncated_normal(shape=[self.num_items], mean=0.0, stddev=0.01),
                                       name='item_biases', dtype=tf.float32)  # (items)

        Gate_GCN_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        self.weights.setdefault("GCN_u_w1", tf.Variable(Gate_GCN_initializer([self.embedding_size, self.embedding_size]),
                                                        dtype=tf.float32))
        self.weights.setdefault("GCN_u_w2", tf.Variable(Gate_GCN_initializer([self.embedding_size, self.embedding_size]),
                                                        dtype=tf.float32))
        self.weights.setdefault("GCN_i_w1", tf.Variable(Gate_GCN_initializer([self.embedding_size, self.embedding_size]),
                                                        dtype=tf.float32))
        self.weights.setdefault("GCN_i_w2", tf.Variable(Gate_GCN_initializer([self.embedding_size, self.embedding_size]),
                                                        dtype=tf.float32))

        # Gate embedding
        Gating_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        self.weights.setdefault("item_gating_w3",
                                tf.Variable(Gating_initializer([self.embedding_size, self.embedding_size]), dtype=tf.float32))
        self.weights.setdefault("item_gating_w4",
                                tf.Variable(Gating_initializer([self.embedding_size, self.embedding_size]), dtype=tf.float32))
        self.weights.setdefault("item_gating_w5",
                                tf.Variable(Gating_initializer([self.embedding_size, self.embedding_size]), dtype=tf.float32))

        self.weights.setdefault("social_gating_w6",
                                tf.Variable(Gating_initializer([self.embedding_size, self.embedding_size]), dtype=tf.float32))
        self.weights.setdefault("social_gating_w7",
                                tf.Variable(Gating_initializer([self.embedding_size, self.embedding_size]), dtype=tf.float32))

        self.user_embeddings, self.target_item_embeddings = self._create_gcn_embed()

    def _create_inference(self):
        # embedding lookup
        self.batch_size_b = tf.shape(self.item_seq_ph)[0]

        self.user_embs = tf.nn.embedding_lookup(self.user_embeddings, self.user_ph)  # (b, d)
        self.social_embs = tf.nn.embedding_lookup(self.social_embeddings, self.social_user_input)  # (b,f, d)
        user_embs = tf.expand_dims(self.user_embs, axis=1)  # (b, 1, d)

        self.tar_item_emb_pos = tf.nn.embedding_lookup(self.target_item_embeddings, self.item_pos_ph)  # b,d
        self.tar_item_emb_neg = tf.nn.embedding_lookup(self.target_item_embeddings, self.item_neg_ph)  # b,d

        self.seq_item_embs = tf.nn.embedding_lookup(self.embeddings["seq_item_embeddings"],
                                                    self.item_seq_ph)  # (b, L, d)

        user_embs = tf.tile(user_embs, tf.stack([1, self.seq_L, 1]))  # b*L*d

        item_gating_output = tf.matmul(tf.reshape(user_embs, [-1, self.embedding_size]), self.weights["item_gating_w3"]) + \
                             tf.matmul(tf.reshape(self.seq_item_embs, [-1, self.embedding_size]),
                                       self.weights["item_gating_w4"])

        item_gating_output = tf.reshape(item_gating_output, [-1, self.seq_L, self.embedding_size])  # b*L*d

        item_gating_output = tf.nn.sigmoid(
            item_gating_output + tf.matmul(self.embeddings["position_embeddings"], self.weights["item_gating_w5"]))

        item_level = tf.reduce_sum(tf.multiply(self.seq_item_embs, item_gating_output), axis=1, keepdims=True)  # b*1*d

        bat_social = tf.shape(self.social_embs)[1]
        item_level_ = tf.tile(item_level, tf.stack([1, bat_social, 1]))

        social_gating_output = tf.matmul(tf.reshape(item_level_, [-1, self.embedding_size]),
                                         self.weights["social_gating_w6"]) + \
                               tf.matmul(tf.reshape(self.social_embs, [-1, self.embedding_size]),
                                         self.weights["social_gating_w7"])

        social_gating_output = tf.reshape(social_gating_output, [-1, bat_social, self.embedding_size])  # b*L*d

        social_gating_output = tf.nn.sigmoid(social_gating_output)

        social_level = tf.reduce_sum(tf.multiply(self.social_embs, social_gating_output), axis=1)  # b*d

        short_term = tf.squeeze(item_level, axis=1) + social_level  # b*d

        self.long_short = tf.tile(self.global_embedding, tf.stack([self.batch_size_b, 1])) + self.user_embs + short_term

        self.item_bias = tf.nn.embedding_lookup(self.item_biases, self.item_pos_ph)
        self.item_bias_neg = tf.nn.embedding_lookup(self.item_biases, self.item_neg_ph)  # b

        self.output = self.item_bias - tf.reduce_sum(tf.square(self.long_short - self.tar_item_emb_pos), 1)
        self.output_neg = self.item_bias_neg - tf.reduce_sum(tf.square(self.long_short - self.tar_item_emb_neg), 1)



    def _create_loss(self):
        self._create_inference()
        self.loss = -tf.reduce_sum(tf.log_sigmoid(self.output - self.output_neg))

        self.L2_emb = tf.reduce_sum(tf.square(self.user_embs)) + tf.reduce_sum(tf.square(self.tar_item_emb_pos)) + \
                      tf.reduce_sum(tf.square(self.tar_item_emb_neg)) + tf.reduce_sum(tf.square(self.seq_item_embs)) + \
                      tf.reduce_sum(tf.square(self.global_embedding)) + tf.reduce_sum(tf.square(self.social_embs)) + \
                      tf.reduce_sum(tf.square(self.item_bias)) + tf.reduce_sum(tf.square(self.item_bias_neg))

        self.L2_weight = tf.reduce_sum(tf.square(self.weights["GCN_u_w1"])) + \
                         tf.reduce_sum(tf.square(self.weights["GCN_u_w2"])) + \
                         tf.reduce_sum(tf.square(self.weights["GCN_i_w1"])) + \
                         tf.reduce_sum(tf.square(self.weights["GCN_i_w2"])) + \
                         tf.reduce_sum(tf.square(self.weights["item_gating_w3"])) + \
                         tf.reduce_sum(tf.square(self.weights["item_gating_w4"])) + \
                         tf.reduce_sum(tf.square(self.weights["item_gating_w5"])) + \
                         tf.reduce_sum(tf.square(self.embeddings["position_embeddings"])) + \
                         tf.reduce_sum(tf.square(self.weights["social_gating_w6"])) + \
                         tf.reduce_sum(tf.square(self.weights["social_gating_w7"]))

        self.loss += self.l2_reg * self.L2_emb + self.l2_W * self.L2_weight

    def _create_optimizer(self):
        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def build_graph(self):
        self._create_placeholder()
        self._create_variable()
        self._create_loss()
        self._create_optimizer()
        pre_emb = tf.expand_dims(self.long_short, axis=1)  # b*1*d
        j_emb = tf.expand_dims(self.target_item_embeddings, axis=0)  # 1*n*d
        self.prediction = -l2_distance(pre_emb, j_emb) + self.item_biases  # b*n

    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        self.user_pos_train = csr_to_user_dict_bytime(self.dataset.time_matrix, self.dataset.train_matrix)
        users_list, item_seq_list, item_pos_list, user_social_list = self._generate_sequences()
        for epoch in range(self.epochs):
            item_neg_list = self._sample_negative(users_list)
            data = DataIterator(users_list, item_seq_list, item_pos_list, item_neg_list, user_social_list,
                                batch_size=self.batch_size, shuffle=True)
            for bat_user, bat_item_seq, bat_item_pos, bat_item_neg, bat_user_social in data:
                bat_user_social = pad_sequences(bat_user_social, value=self.num_users)
                feed = {self.user_ph: bat_user,
                        self.item_seq_ph: bat_item_seq,
                        self.item_pos_ph: bat_item_pos,
                        self.item_neg_ph: bat_item_neg,
                        self.social_user_input: bat_user_social}
                # self.sess.run([self.train_opt, self.minimize_nn], feed_dict=feed)
                self.sess.run(self.train_opt, feed_dict=feed)

            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))

    def _generate_sequences(self):
        self.user_test_seq = {}
        user_list, item_seq_list, item_pos_list, user_social_list = [], [], [], []

        for user_id in range(self.num_users):
            seq_items = self.user_pos_train[user_id]
            social_friends = self.SocialtrainDict[user_id]
            for index_id in range(len(seq_items)):
                content_data = list()
                for cindex in range(max([0, index_id - self.seq_L]), index_id):
                    content_data.append(seq_items[cindex])

                if (len(content_data) < self.seq_L):
                    content_data = content_data + [self.num_items for _ in range(self.seq_L - len(content_data))]
                user_list.append(user_id)
                item_seq_list.append(content_data)
                item_pos_list.append(seq_items[index_id])
                user_social_list.append(social_friends)
            user_id_seq = seq_items[-min([len(seq_items), self.seq_L]):]
            if (len(seq_items) < self.seq_L):
                user_id_seq = user_id_seq + [self.num_items for _ in range(self.seq_L - len(user_id_seq))]
            self.user_test_seq[user_id] = user_id_seq

        return user_list, item_seq_list, item_pos_list, user_social_list

    def _sample_negative(self, users_list):
        neg_items_list = []
        user_neg_items_dict = {}
        all_uni_user, all_counts = np.unique(users_list, return_counts=True)
        user_count = DataIterator(all_uni_user, all_counts, batch_size=1024, shuffle=False)
        for bat_users, bat_counts in user_count:
            n_neg_items = [c for c in bat_counts]
            exclusion = [self.user_pos_train[u] for u in bat_users]
            bat_neg = batch_randint_choice(self.num_items, n_neg_items, replace=True, exclusion=exclusion)
            for u, neg in zip(bat_users, bat_neg):
                user_neg_items_dict[u] = neg

        for u, c in zip(all_uni_user, all_counts):
            neg_items = np.reshape(user_neg_items_dict[u], newshape=[c, ])
            neg_items_list.extend(neg_items)
        return neg_items_list

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, user_ids, candidate_items_userids=None):
        ratings = []
        if candidate_items_userids is not None:
            for userid, itemids in zip(user_ids, candidate_items_userids):
                item_recent, user_social = [], []
                recents = self.user_test_seq[userid]
                friends = self.SocialtrainDict[userid]
                for _ in range(len(itemids)):
                    item_recent.append(recents)
                    user_social.append(friends)
                users = np.full(len(itemids), userid, dtype=np.int32)
                feed_dict = {self.user_ph: users,
                             self.item_pos_ph: itemids,
                             self.item_seq_ph: item_recent,
                             self.social_user_input: user_social}
                ratings.append(self.sess.run(self.output, feed_dict=feed_dict))
        return ratings