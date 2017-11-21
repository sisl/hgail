
import tensorflow as tf

def _build_dense_module(
        name,
        inputs, 
        hidden_layer_dims,
        activation_fn=tf.nn.relu,
        bias_initializer=tf.constant_initializer(0.1),
        dropout_keep_prob=1.,
        weights_regularizer=None):
    with tf.variable_scope(name):
        hidden = inputs
        for hidden_dim in hidden_layer_dims:
            hidden = tf.contrib.layers.fully_connected(
                        hidden, 
                        hidden_dim, 
                        activation_fn=activation_fn,
                        biases_initializer=bias_initializer,
                        weights_regularizer=weights_regularizer)
            hidden = tf.nn.dropout(hidden, dropout_keep_prob)
        return hidden

def _build_score_module(
        name, 
        hidden, 
        output_dim=1, 
        weights_regularizer=None):
    with tf.variable_scope(name):
        scores = tf.contrib.layers.fully_connected(
                    hidden, 
                    output_dim, 
                    activation_fn=None, 
                    biases_initializer=tf.constant_initializer(0.0),
                    weights_regularizer=weights_regularizer)
        return scores

class Network(object):
    
    def __init__(self, name):
        self.name = name
        self.reuse = False
    
    def __call__(self, *inputs, **kwargs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            outputs = self._build(*inputs, **kwargs)
            
        if self.reuse == False:
            self.reuse = True
            self.outputs = outputs
            self.inputs = inputs
            self.var_list = [v for v in tf.trainable_variables() if self.name in v.name]
            
        return outputs 

    def _build(self):
        raise NotImplementedError()
    
    def _run(self, inputs, extra_feed=dict()):
        feed_dict = {var: val for (var, val) in zip(self.inputs, inputs)}
        feed_dict.update(extra_feed)
        session = tf.get_default_session()
        outputs = session.run(self.outputs, feed_dict=feed_dict)
        return outputs

    def get_param_values(self):
        session = tf.get_default_session()
        return [session.run(v) for v in self.var_list]

    def set_param_values(self, values):
        assign = tf.group(*[tf.assign(var, val) for (var, val) in zip(self.var_list, values)])
        session = tf.get_default_session()
        session.run(assign)

class Classifier(Network):
    
    def __init__(
            self,
            name,
            hidden_layer_dims,
            output_dim=1,
            activation_fn=tf.nn.relu,
            dropout_keep_prob=1.,
            l2_reg=0.,
            **kwargs):
        super(Classifier, self).__init__(name=name, **kwargs)
        self.name = name
        self.hidden_layer_dims = hidden_layer_dims
        self.output_dim = output_dim
        self.activation_fn = activation_fn
        self.dropout_keep_prob = dropout_keep_prob
        self.dropout_keep_prob_ph = tf.placeholder_with_default(
            self.dropout_keep_prob, 
            shape=(), 
            name='dropout_keep_prob_ph'
        )
        self.l2_reg = l2_reg

    def _build(self, inputs):
        hidden = _build_dense_module(
            '{}/hidden'.format(self.name),
            inputs, 
            self.hidden_layer_dims,
            activation_fn=self.activation_fn,
            dropout_keep_prob=self.dropout_keep_prob_ph,
            weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        # score
        score = _build_score_module(
            '{}/scores'.format(self.name), 
            hidden, 
            output_dim=self.output_dim,
            weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )      
        return score

    def forward(self, inputs, deterministic=False):
        extra_feed = {self.dropout_keep_prob_ph: 1.} if deterministic else {}
        return self._run([inputs], extra_feed=extra_feed)   

class ObservationActionMLP(Network):

    def __init__(
            self,
            name,
            hidden_layer_dims,
            output_dim=1,
            obs_hidden_layer_dims=[],
            act_hidden_layer_dims=[],
            activation_fn=tf.nn.relu,
            dropout_keep_prob=1.,
            l2_reg=0.,
            return_features=False,
            **kwargs):
        super(ObservationActionMLP, self).__init__(name=name, **kwargs)
        self.output_dim = output_dim
        self.obs_hidden_layer_dims = obs_hidden_layer_dims
        self.act_hidden_layer_dims = act_hidden_layer_dims
        self.hidden_layer_dims = hidden_layer_dims
        self.activation_fn = activation_fn
        self.dropout_keep_prob = dropout_keep_prob
        self.return_features = return_features
        self.dropout_keep_prob_ph = tf.placeholder_with_default(
            self.dropout_keep_prob, 
            shape=(), 
            name='dropout_keep_prob_ph'
        )
        self.l2_reg = l2_reg

    def _build(self, obs, act):
        # obs
        obs_hidden = _build_dense_module(
            '{}/obs'.format(self.name), 
            obs, 
            self.obs_hidden_layer_dims, 
            activation_fn=self.activation_fn,
            dropout_keep_prob=self.dropout_keep_prob_ph,
            weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        # act
        act_hidden = _build_dense_module(
            '{}/act'.format(self.name), 
            act, 
            self.act_hidden_layer_dims,
            activation_fn=self.activation_fn,
            dropout_keep_prob=self.dropout_keep_prob_ph,
            weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )
        # hidden layers
        hidden = tf.concat([obs_hidden, act_hidden], axis=1)
        features = [hidden]
        with tf.variable_scope('{}/hidden'.format(self.name)):
            for hidden_dim in self.hidden_layer_dims:
                hidden = tf.contrib.layers.fully_connected(
                            hidden, 
                            hidden_dim,
                            activation_fn=self.activation_fn,
                            biases_initializer=tf.constant_initializer(0.1),
                            weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg))
                features.append(hidden)
                hidden = tf.nn.dropout(hidden, self.dropout_keep_prob_ph)

        # score
        score = _build_score_module(
            '{}/scores'.format(self.name), 
            hidden, 
            output_dim=self.output_dim,
            weights_regularizer=tf.contrib.layers.l2_regularizer(self.l2_reg)
        )         

        if self.return_features:
            return score, features
        else:       
            return score
    
    def forward(self, obs, act, deterministic=False):
        extra_feed = {self.dropout_keep_prob_ph: 1.} if deterministic else {}
        return self._run([obs, act], extra_feed=extra_feed)

def CriticNetwork(**kwargs):
    return ObservationActionMLP(name='critic', **kwargs)

class ConvPredictor(Network):
    
    def __init__(
            self, 
            name,
            output_dim=1, 
            n_layers=5,
            n_filters=[32,64,64,64,64],
            dense_hidden_dim=256,
            dropout_keep_prob=1.,
            **kwargs):
        super(ConvPredictor, self).__init__(name=name, **kwargs)
        self.name = name
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.dense_hidden_dim = dense_hidden_dim
        self.dropout_keep_prob = dropout_keep_prob

        # note that this is created here because _build may be called 
        # multiple times, which can cause issues with this variable
        # ph for dropout keep prob to allow for switching it
        # default ph just passes through dropout_keep_prob if a value 
        # if not explicitly feed for it
        self.dropout_keep_prob_ph = tf.placeholder_with_default(
            self.dropout_keep_prob, 
            shape=(), 
            name='dropout_keep_prob_ph'
        )
        
    def _build(self, inputs, extra_inputs=None):

        hidden = inputs
        for i in range(self.n_layers):
            hidden = tf.layers.conv2d(
                hidden,
                filters=self.n_filters[i],
                kernel_size=3,
                strides=(2,2),
                activation=tf.nn.relu,
                bias_initializer=tf.constant_initializer(0.1))
            hidden = tf.nn.dropout(hidden, self.dropout_keep_prob_ph)   

        hidden = tf.contrib.layers.flatten(hidden)

        # bring in extra inputs if provided
        if extra_inputs is not None:
            hidden = tf.concat((hidden, extra_inputs), axis=-1)
            # extra couple hidden layers for good measure
            for _ in range(2):
                hidden = tf.contrib.layers.fully_connected(
                        hidden, 
                        self.dense_hidden_dim, 
                        activation_fn=tf.nn.relu, 
                        biases_initializer=tf.constant_initializer(0.1))
                hidden = tf.nn.dropout(hidden, self.dropout_keep_prob_ph)

        hidden = tf.contrib.layers.fully_connected(
                    hidden, 
                    self.dense_hidden_dim, 
                    activation_fn=tf.nn.relu, 
                    biases_initializer=tf.constant_initializer(0.1))
        outputs = tf.contrib.layers.fully_connected(
                    hidden, 
                    self.output_dim, 
                    activation_fn=None, 
                    biases_initializer=tf.constant_initializer(0.))

        return outputs
    
    def forward(self, inputs, extra_inputs=None, deterministic=False):
        if deterministic:
            extra_feed = {self.dropout_keep_prob_ph: 1.}
        else:
            # don't pass anything because using default placeholders
            extra_feed = {}

        inputs_list = [inputs]
        if extra_inputs is not None:
            inputs_list += [extra_inputs]
        
        return self._run(inputs_list, extra_feed)
