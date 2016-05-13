//
//  main.cpp
//  Neurons
//
//  Created by Steven Vayl on 5/13/16.
//  Copyright (c) 2016 Steven Vayl. All rights reserved.
//
// neural-net-tutorial.cpp

#include  <vector>
#include <iostream>

using namespace std;

class Neuron {} ;

typedef vector<Neuron> Layer;

class Net
{
public:
	Net(const vector<unsigned> &topology);    //const insures we do not change the topology of the network.
	void feedForward(const vector<double> &inputVals) {};
	void backProp(const vector<double> &targetVals) {};
	void getResults(vector<double> &resultVals) const {};   //(const qualifier)
    
private:
    vector<Layer> m_layers;   //m_layers[layerNum][neuronNum]
};

Net::Net(const vector<unsigned> &topology)
{
    unsigned numLayers = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum) {
        m_layers.push_back(Layer());
        
        // we have made a new Layer, now fill it with neurons, and add a bias neuron to the layer
        
        for(unsigned neuronNum =0; neuronNum <= topology[layerNum]; ++neuronNum) {
            m_layers.back().push_back(Neuron());
            cout << "Made a Neuron!" << endl;
		}
	}
}

int main()
{
    // e.g.. (3, 2, 1) -> 3 neurons in first layer, 2 in middle layer, 1 in final (output) layer.
    vector<unsigned>  topology;
    
    //creating the network
    topology.push_back(3); // 3 neurons in first layer +bias neuron
    topology.push_back(2); // 2 neurons in second layer +bias neuron
    topology.push_back(1); // 1 neuron in final layer +bias neuron
    
    
	Net myNet(topology);  //constructor
    
	
    
    vector<double> inputVals;
    myNet.feedForward(inputVals);    //member function
    
    vector<double> targetVals;
    myNet.backProp(targetVals);
    
    vector<double> resultVals;
	myNet.getResults(resultVals);
}
