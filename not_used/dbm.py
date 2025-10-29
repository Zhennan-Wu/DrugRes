import sklearn
from utils import binarize
from sklearn.pipeline import Pipeline


class DBM:

    def __init__(self, dbn_pipeline: Pipeline):

        self.layer_types =  [step[0] for step in dbn_pipeline.steps]
        layer_batch_size = [step.get_params()['batch_size'] for step in dbn_pipeline.steps]
        if (len(set(layer_batch_size)) == 1):
            self.batch_size = layer_batch_size[0]
        else:
            raise ValueError("Batch sizes of all layers must be the same")
        
        self.W = [step[1].components_ for step in dbn_pipeline.steps]
        self.num_layers = len(self.W)

        self.hidden_bias = []
        layer_hidden_biases = [step[1].intercept_hidden_ for step in dbn_pipeline.steps]
        layer_visible_biases = [step[1].intercept_visible_ for step in dbn_pipeline.steps]
        layer_visible_biases.append(layer_hidden_biases[-1])
        for layer in range(self.num_layers):
            self.hidden_bias.append((layer_hidden_biases[layer] + layer_visible_biases[layer+1])/2.)
        self.visible_bias = layer_visible_biases[0]
    
    def init_markov_chain(self, data, dbn_pipeline):
        """
        Initialize the Markov chain with the first layer of the DBN.
        """
        self.markov_chain = [data]
        rbm_layers = [step[1] for step in dbn_pipeline.steps]
        for layer_idx, rbm in enumerate(rbm_layers):
            if not "gaussian" in self.layer_types[layer_idx]:
                input_data = binarize(input_data, input_data.mean())
                "Input data binarized"
            energy_mean_tracking = []
            energy_var_tracking = []

            for epoch in range(epochs):
                energy_epoch = []
                # Shuffle the full data for each epoch

                input_data, input_labels = shuffle(input_data, input_labels, random_state=layer_idx+epoch)


                for i in range(0, len(input_data), rbm.batch_size):
                    batch = input_data[i:i + rbm.batch_size]
                    if batch.shape[0] < rbm.batch_size:
                        continue  # Skip incomplete batch

                    rbm.partial_fit(batch)

                    if hasattr(rbm, 'score_samples'):
                        energy = -rbm.score_samples(batch)
                        energy_epoch.extend(energy)

                energy_epoch = np.array(energy_epoch)
                energy_mean_tracking.append(energy_epoch.mean())
                energy_var_tracking.append(energy_epoch.var())

            energy_tracking_per_layer.append((energy_mean_tracking, energy_var_tracking))

            # Transform data for next layer
            input_data = rbm.transform(input_data)

    def model(self, data):
        markov_chain = [data]
        for i in range(self.num_layers):
            markov_chain.append(pyro.sample(f"hidden_{i+1}", torch.matmul(markov_chain[i], self.W[i].T)))

    def guide(self):
        pass
