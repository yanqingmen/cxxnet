/*
 * rnn_layer-inl.hpp
 * simple rnn layer.
 *  Created on: 2015年6月29日
 *      Author: hzx
 */

#ifndef CXXNET_LAYER_RNN_LAYER_INL_HPP_
#define CXXNET_LAYER_RNN_LAYER_INL_HPP_
#include <mshadow/tensor.h>
#include "./layer.h"
#include "./param.h"
#include "./op.h"
#include "../utils/utils.h"

namespace cxxnet {
namespace layer {

/* !
 * \breif a simple rnn layer, not quite useful in practical, for experimental purpose
 */
template<typename xpu>
class SimpleRNNLayer : public ILayer<xpu> {
public:
SimpleRNNLayer(mshadow::Random<xpu> *p_rnd) : bptt(6), size(0), end(-1), clip_value(20.0) {
	fc_layer = new FullConnectLayer<xpu>(p_rnd);
	act_layer = new ActivationLayer<xpu, op::sigmoid, op::sigmoid_grad>();
	tmp_nodes.push_back(new Node<xpu>());
}

virtual ~SimpleRNNLayer(void) {
	delete fc_layer;
	delete act_layer;
	for(int i=0; i<tmp_nodes.size(); i++) {
		delete tmp_nodes[i];
	}
}

virtual void SetParam(const char *name, const char* val) {
	if (!strcmp(name, "bptt")){
		bptt = atoi(val);
		utils::Check(bptt > 0, "SimpleRNNLayer: bptt should be greater than 0");
	}
	if (!strcmp(name, "clip")){
		clip_value = atof(val);
		utils::Check(clip_value > 0, "SimpleRNNLayer: clip_value should be greater than 0");
	}
	else{
		fc_layer->SetParam(name, val);
	}
}

virtual void ApplyVisitor(typename ILayer<xpu>::IVisitor *pvisitor) {
	fc_layer->ApplyVisitor(pvisitor);
}

virtual void InitModel(void) {
	fc_layer->InitModel();
}

virtual void SaveModel(utils::IStream &fo) const {
	fc_layer->SaveModel(fo);
}

virtual void LoadModel(utils::IStream &fi) {
	fc_layer->LoadModel(fi);
}

virtual void SetStream(mshadow::Stream<xpu> *stream) {
	fc_layer->SetStream(stream);
}

virtual void InitConnection(const std::vector<Node<xpu>*> &nodes_in,
                              const std::vector<Node<xpu>*> &nodes_out,
                              ConnectState<xpu> *p_cstate) {
	Node<xpu>* tnode = nodes_out[0];
	//all nodes in and out should be with same shape
	for(int i=0; i<nodes_in.size(); i++) {
		utils::Check(nodes_in[i]->is_mat(), "SRNN: input need to be a matrix");
	}
	tnode->data.shape_ =
	        mshadow::Shape4(nodes_in[0]->data.size(0), 1, 1, nodes_in[0]->data.size(3));
	//used to save history states for bptt
	p_cstate->states.resize(3);
	p_cstate->states[0].Resize(mshadow::Shape4(tnode->data.size(0), 1, 1, tnode->data.size(3)));
	p_cstate->states[1].Resize(mshadow::Shape4(1, bptt, tnode->data.size(0), tnode->data.size(3)));
	p_cstate->states[2].Resize(mshadow::Shape4(1, bptt, tnode->data.size(0), tnode->data.size(3)));
	//for the inner recurrent layer, input_node is the same as output_node
	tmp_nodes[0]->data = p_cstate->states[0];
	fc_layer->InitConnection(nodes_out, tmp_nodes, p_cstate);
}

virtual void Forward(bool is_train,
                       const std::vector<Node<xpu>*> &nodes_in,
                       const std::vector<Node<xpu>*> &nodes_out,
                       ConnectState<xpu> *p_cstate) {
	using namespace mshadow::expr;
	using namespace cxxnet::op;
	//copy latest state as input
	if(end >=0 ) {
		mshadow::Copy(nodes_out[0]->data.FlatTo2D(), p_cstate->states[2][0][end]);
	}

	fc_layer->Forward(is_train, nodes_out, tmp_nodes, p_cstate);
	//combine all input nodes
	for(int i=0; i<nodes_in.size(); i++) {
		tmp_nodes[0]->data += nodes_in[i]->data;
	}
	tmp_nodes[0]->data = F<clip>(tmp_nodes[0]->data, clip_value);
	act_layer->Forward(is_train, tmp_nodes, nodes_out, p_cstate);
	this->BackupStates(tmp_nodes, nodes_out, p_cstate);
}

virtual void Backprop(bool prop_grad,
                        const std::vector<Node<xpu>*> &nodes_in,
                        const std::vector<Node<xpu>*> &nodes_out,
                        ConnectState<xpu> *p_cstate) {
	//copy gradient
	for(int i=0; i<nodes_in.size(); i++) {
		mshadow::Copy(nodes_in[i]->data, nodes_out[0]->data);
	}

	//backprob through time
	for(int time=1; time<size; time++) {
		this->LoadTmpState(tmp_nodes, p_cstate, time-1);
		act_layer->Backprop(prop_grad, tmp_nodes, nodes_out,  p_cstate);
		this->LoadHidenState(nodes_out, p_cstate, time);
		fc_layer->Backprop(prop_grad, nodes_out, tmp_nodes, p_cstate);
	}
}

protected:
/*!
 * \brief load states from history for bptt
 * \param time: backprob time, e.g. time=0 means the latest state, time=1 means the state previous latest
 */
virtual void LoadTmpState(const std::vector<Node<xpu>*> &tmp_nodes,
		ConnectState<xpu> *p_cstate, int time) {
	utils::Check(size > time, "time value should be smaller than history size");
	utils::Check(time >= 0, "time should be non-negative number");
	int state = end - time;
	if(state < 0) state += bptt;
	mshadow::Copy(tmp_nodes[0]->data.FlatTo2D(), p_cstate->states[1][0][state]);
}

virtual void LoadHidenState(const std::vector<Node<xpu>*> &nodes_out,
		ConnectState<xpu> *p_cstate, int time) {
	utils::Check(size > time, "time value should be smaller than history size");
	utils::Check(time >= 0, "time should be non-negative number");
	int state = end - time;
	if(state < 0) state += bptt;
	mshadow::Copy(nodes_out[0]->data.FlatTo2D(), p_cstate->states[2][0][state]);
}

//backup states for bptt
virtual void BackupStates(const std::vector<Node<xpu>*> &tmp_nodes,
		const std::vector<Node<xpu>*> &nodes_out,
		ConnectState<xpu> *p_cstate) {
	end++;
	size++;
	end = end%bptt;
	if(size>bptt) size=bptt;
	mshadow::Copy(p_cstate->states[1][0][end], tmp_nodes[0]->data.FlatTo2D());
	mshadow::Copy(p_cstate->states[2][0][end], nodes_out[0]->data.FlatTo2D());
}
	/*! a simple rnn layer could be constructed by combine a full-connect layer and a sigmoid active layer */
	ILayer<xpu>* fc_layer;
	ILayer<xpu>* act_layer;
	/*! tmp_node for cal utility */
	std::vector<Node<xpu>*> tmp_nodes;
	/*! \brief parameters that potentially be useful */
//	LayerParam param_;
	/*! size for back prop */
	int bptt;
	/*! indexes for bptt */
	int size, end;
	//clip value
	float clip_value;
};


} // namespace layer
} // namespace cxxnet





#endif /* SRC_LAYER_RNN_LAYER_INL_HPP_ */
