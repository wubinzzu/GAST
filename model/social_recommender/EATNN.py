'''
Reference: Tong Zhao et al., "Leveraging Social Connections to Improve 
Personalized Ranking for Collaborative Filtering." in CIKM 2014
@author: wubin
'''
import tensorflow as tf
import numpy as np
from time import time
from util import learner
from model.AbstractRecommender import SocialAbstractRecommender
from util.data_iterator import DataIterator
from util.tool import pad_sequences, csr_to_user_dict, timer
#from boto.dynamodb2 import items

class EATNN(SocialAbstractRecommender):
    def __init__(self, sess, dataset, conf):
        super(EATNN, self).__init__(dataset, conf)
        self.learning_rate = conf["learning_rate"]
        self.embedding_size = conf["embedding_size"]
        self.attention_size= conf["attention_size"]
        self.learner = conf["learner"]
        self.num_epochs= conf["epochs"]
        self.batch_size = conf["batch_size"]
        self.verbose = conf["verbose"]
        self.keep_prob = conf["keep_prob"]
        self.alpha = conf["alpha"]
        self.mu = conf["mu"]
        self.lambda_bilinear = [1e-3,1e-1, 1e-2]
        self.dataset = dataset
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.SocialtrainDict = self._get_SocialDict()
        self.train_dict = csr_to_user_dict(self.dataset.train_matrix)
        self.sess = sess

    def _get_SocialDict(self):
        #find items rated by trusted neighbors only
        SocialDict = {}
        for u in range(self.num_users):
            trustors = self.social_matrix[u].indices
            if len(trustors)>0:
                SocialDict[u] = trustors.tolist()
            else:
                SocialDict[u] = []
        return SocialDict

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, [None,], name="user_input")
    
            self.items_input =tf.placeholder(tf.int32, [None, None], name="items_input")
            self.friends_input = tf.placeholder(tf.int32, [None, None], name="friends_input")
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            
    def _create_variables(self):
        self.uidW_g = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="uidWg")
        self.uidW_i = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0,
                                                      stddev=0.01), dtype=tf.float32, name="uidWi")
        self.uidW_s = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0,
                                                      stddev=0.01), dtype=tf.float32, name="uidWs")
       
        self.c1 = tf.Variable(tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="c1")
        self.c2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='c2')
        
        self.iidW = tf.concat([self.c1,self.c2], 0, name='iidW')
        
        
        self.s1 = tf.Variable(tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0,
                                                    stddev=0.01), dtype=tf.float32, name="s1")
        self.s2 = tf.constant(0.0, tf.float32, [1, self.embedding_size], name='s2')
        
        self.fidW = tf.concat([self.s1,self.s2], 0, name='fidW')

        # item domain
        self.H_i = tf.Variable(tf.constant(0.01, shape=[self.embedding_size, 1]), name="hi")

        # social domain
        self.H_f = tf.Variable(tf.constant(0.01, shape=[self.embedding_size, 1]), name="hf")

        # item domain attention
        self.WA = tf.Variable(
            tf.truncated_normal(shape=[self.embedding_size, self.attention_size], mean=0.0, stddev=tf.sqrt(
                tf.div(2.0, self.attention_size + self.embedding_size))), dtype=tf.float32, name='WA')
        self.BA = tf.Variable(tf.constant(0.00, shape=[self.attention_size]), name="BA")
        self.HA = tf.Variable(tf.constant(0.01, shape=[self.attention_size, 1]), name="HA")

        # social domain attention
        self.WB = tf.Variable(
            tf.truncated_normal(shape=[self.embedding_size, self.attention_size], mean=0.0, stddev=tf.sqrt(
                tf.div(2.0, self.attention_size + self.embedding_size))), dtype=tf.float32, name='WB')
        self.BB = tf.Variable(tf.constant(0.00, shape=[self.attention_size]), name="BB")
        self.HB = tf.Variable(tf.constant(0.01, shape=[self.attention_size, 1]), name="HB")
    
    def _item_attentive_transfer(self):

        item_w=tf.exp(tf.matmul(tf.nn.relu(tf.matmul(self.uid_i,self.WA)+self.BA),self.HA))
        general_w=tf.exp(tf.matmul(tf.nn.relu(tf.matmul(self.uid_g,self.WA)+self.BA),self.HA))
        item_w=tf.div(item_w,item_w+general_w)
        general_w=1.0-item_w
        uid_A=item_w*self.uid_i + general_w*self.uid_g
        return uid_A,item_w

    def _social_attentive_transfer(self):

        social_w = tf.exp(tf.matmul(tf.nn.relu(tf.matmul(self.uid_s, self.WB) + self.BB), self.HB))
        general_w = tf.exp(tf.matmul(tf.nn.relu(tf.matmul(self.uid_g, self.WB) + self.BB), self.HB))
        social_w = tf.div(social_w, social_w + general_w)
        general_w = 1.0 - social_w
        uid_B = social_w * self.uid_s + general_w * self.uid_g
        return uid_B,social_w
        
    def _create_inference(self):
        with tf.name_scope("inference"):
            self.uid_g = tf.nn.embedding_lookup(self.uidW_g, self.user_input)
            self.uid_i = tf.nn.embedding_lookup(self.uidW_i, self.user_input)
            self.uid_s = tf.nn.embedding_lookup(self.uidW_s, self.user_input)
    
            self.uid_A,self.item_w=self._item_attentive_transfer()
            self.uid_B,self.social_w=self._social_attentive_transfer()
    
            self.uid_A = tf.nn.dropout(self.uid_A, self.dropout_keep_prob)
            self.uid_B = tf.nn.dropout(self.uid_B, self.dropout_keep_prob)
    
            self.pos_item = tf.nn.embedding_lookup(self.iidW,self.items_input)
            self.pos_num_r = tf.cast(tf.not_equal(self.items_input, self.num_items), 'float32')
            self.pos_item = tf.einsum('ab,abc->abc', self.pos_num_r, self.pos_item)
            self.pos_r=tf.einsum('ac,abc->abc',self.uid_A,self.pos_item)
            self.pos_r=tf.einsum('ajk,kl->ajl', self.pos_r, self.H_i)
            self.pos_r = tf.reshape(self.pos_r, [-1, tf.shape(self.pos_item)[1]])

            self.pos_friend = tf.nn.embedding_lookup(self.fidW, self.friends_input)
            self.pos_num_f = tf.cast(tf.not_equal(self.friends_input, self.num_users), 'float32')
            self.pos_friend = tf.einsum('ab,abc->abc', self.pos_num_f, self.pos_friend)
            self.pos_f = tf.einsum('ac,abc->abc', self.uid_B, self.pos_friend)
            self.pos_f = tf.einsum('ajk,kl->ajl', self.pos_f, self.H_f)
            self.pos_f = tf.reshape(self.pos_f, [-1, tf.shape(self.pos_friend)[1]])

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.loss1=self.alpha*tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.einsum('ab,ac->abc',self.iidW,self.iidW),0)
                                    *tf.reduce_sum(tf.einsum('ab,ac->abc',self.uid_A,self.uid_A),0)
                                    *tf.matmul(self.H_i,self.H_i,transpose_b=True),0),0)
            self.loss2=self.alpha*tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.einsum('ab,ac->abc',self.fidW,self.fidW),0)
                                        *tf.reduce_sum(tf.einsum('ab,ac->abc',self.uid_B,self.uid_B),0)
                                        *tf.matmul(self.H_f,self.H_f,transpose_b=True),0),0)
    
            self.loss1+=tf.reduce_sum((1.0 - self.alpha) * tf.square(self.pos_r) - 2.0 * self.pos_r)
            self.loss2+=tf.reduce_sum((1.0-self.alpha)*tf.square(self.pos_f)-2.0*self.pos_f)
    
            self.l2_loss0=tf.nn.l2_loss(self.uid_A+self.uid_B)
    
            self.l2_loss1 = tf.nn.l2_loss(self.WA) + tf.nn.l2_loss(self.BA)+tf.nn.l2_loss(self.HA)
            self.l2_loss2 = tf.nn.l2_loss(self.WB) + tf.nn.l2_loss(self.BB)+tf.nn.l2_loss(self.HB)
    
            self.loss=self.loss1+self.mu*self.loss2\
                      +self.lambda_bilinear[0]*self.l2_loss0\
                      +self.lambda_bilinear[1]*self.l2_loss1\
                      +self.lambda_bilinear[2]*self.l2_loss2
    def _create_optimizer(self):
        with tf.name_scope("learner"):
            self.optimizer = learner.optimizer(self.learner, self.loss, self.learning_rate)
    
    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_inference()
        self._create_loss()
        self._create_optimizer()
    
#---------- training process -------
    def train_model(self):
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.num_epochs):
            user_train, item_train, friend_train = self._get_input_data()
            data_iter = DataIterator(user_train, item_train, friend_train,
                                     batch_size=self.batch_size, shuffle=True)
            total_loss = 0.0
            training_start_time = time()
            num_training_instances = len(user_train)
            for bat_user_train, bat_item_train, bat_friend_train, in data_iter:
                bat_item_train = pad_sequences(bat_item_train, value=self.num_items)
                bat_friend_train = pad_sequences(bat_friend_train, value=self.num_users)

                feed_dict = {self.user_input: bat_user_train,
                             self.items_input: bat_item_train,
                             self.friends_input: bat_friend_train,
                             self.dropout_keep_prob: self.keep_prob}

                loss, _ = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
                total_loss += loss
            # self.logger.info("[iter %d : loss : %f, time: %f]" % (epoch, total_loss / num_training_instances,
            #                                                       time() - training_start_time))
            if epoch % self.verbose == 0:
                self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))
    
    @timer
    def evaluate(self):
        return self.evaluator.evaluate(self)

    def _get_input_data(self):
        user_train, item_train, friend_train = [], [], []
        for u in range(self.num_users):
            items_by_user = self.train_dict[u]
            friends_by_user =  self.SocialtrainDict[u]
            user_train.append(u)
            item_train.append(items_by_user)
            friend_train.append(friends_by_user)
    
        user_train = np.array(user_train)
        item_train = np.array(item_train)
        friend_train = np.array(friend_train)
        
        num_training_instances = len(user_train)
        shuffle_index = np.arange(num_training_instances,dtype=np.int32)
        np.random.shuffle(shuffle_index)
        user_train = user_train[shuffle_index]
        item_train = item_train[shuffle_index]
        friend_train = friend_train[shuffle_index]
        
        return user_train, item_train, friend_train
            
    def predict(self, user_ids, candidate_items_userids=None):
        ratings = []
        if candidate_items_userids is not None:
            for user_id, items_by_u in zip(user_ids, candidate_items_userids):
                eval_items = np.array(items_by_u)
                eval_items = eval_items[np.newaxis,:]
                friends = np.array(self.SocialtrainDict[user_id])
                friends = friends[np.newaxis,:]
                result = self.sess.run(self.pos_r,
                                   feed_dict={self.user_input: [user_id], 
                                   self.items_input:eval_items,
                                   self.friends_input:friends,
                                   self.dropout_keep_prob:1.0})
                
                ratings.append(np.reshape(result, [-1]))
        else:
            candidate_items_userids = np.arange(self.num_items)
            for user_id in user_ids:
                eval_items = np.array(candidate_items_userids)
                eval_items = eval_items[np.newaxis,:]
                friends = np.array(self.SocialtrainDict[user_id])
                friends = friends[np.newaxis,:]
                result = self.sess.run(self.pos_r,
                                   feed_dict={self.user_input: [user_id], 
                                   self.items_input:eval_items,
                                   self.friends_input:friends,
                                   self.dropout_keep_prob:1.0})   
                ratings.append(np.reshape(result, [-1]))
        return ratings