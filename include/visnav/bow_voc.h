/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <bitset>
#include <cstdlib>
#include <fstream>
#include <unordered_map>

#include <cereal/archives/binary.hpp>

#include <visnav/common_types.h>

namespace cereal {
class access;
}

namespace visnav {

class BowVocabulary {
 public:
  using NodeId = unsigned int;
  using TDescriptor = std::bitset<256>;

  BowVocabulary(const std::string& filename) { load(filename); }

  inline void transformFeatureToWord(const TDescriptor& feature,
                                     WordId& word_id, WordValue& weight) const {
    // TODO SHEET 3: propagate feature through the vocabulary tree stored in the
    // array m_nodes. The root node has id=0 (m_nodes[0]). The array
    // m_nodes[id].children stores ids (index in the array) of the children
    // nodes, m_nodes[id].isLeaf() is true for the leaf nodes.
    // m_nodes[id].descriptor stores the centroid of the node. Start from the
    // root node and propagate to the node that has a smallest distance between
    // feature and descriptor of the cluster centroid. Iterate until you reach
    // the leaf node. Save m_nodes[id].word_id and m_nodes[id].weight of the
    // leaf node to the corresponding variables.
    const Node* n_ptr = &m_nodes[0];  // point the current node pointer to the
                                      // root as initialization
    while (!n_ptr->isLeaf()) {
      // Determine which child has the smallest distance to the feature
      int min_dist = 256;
      NodeId min_id;
      for (auto& child_id : n_ptr->children) {
        int dist = (m_nodes[child_id].descriptor ^ feature).count();
        if (dist < min_dist) {
          min_dist = dist;
          min_id = child_id;
        }
      }
      n_ptr = &m_nodes[min_id];
    }
    word_id = n_ptr->word_id;
    weight = n_ptr->weight;
  }

  inline void transform(const std::vector<TDescriptor>& features,
                        BowVector& v) const {
    v.clear();

    if (m_nodes.empty()) {
      return;
    }

    // TODO SHEET 3: transform the entire vector of features from an image to
    // the BoW representation (you can use transformFeatureToWord function).
    // Use L1 norm to normalize the resulting BoW vector.
    auto exist = [&vec = v](WordId x, WordValue w) {
      bool exist = false;
      for (auto& element : vec) {
        if (element.first == x) {
          element.second += w;
          exist = true;
          break;
        } else {
        }
      }
      if (!exist) {
        vec.push_back(std::make_pair(x, w));
      }
      return exist;
    };

    double L1_norm = 0;
    for (auto& feature : features) {
      WordId id;
      WordValue value;
      transformFeatureToWord(feature, id, value);

      if (value == 0) {
        // Don't add
      } else {
        L1_norm += std::abs(value);
        if (exist(id, value)) {
        } else {
        }
      }
    }
    for (auto& pair : v) {
      pair.second /= L1_norm;
    }
  }

  void save(const std::string& filename) const {
    std::ofstream os(filename, std::ios::binary);

    if (os.is_open()) {
      cereal::BinaryOutputArchive archive(os);

      archive(*this);

    } else {
      std::cout << "Failed to save vocabulary as " << filename << std::endl;
    }
  }

  void load(const std::string& filename) {
    std::ifstream is(filename, std::ios::binary);

    if (is.is_open()) {
      cereal::BinaryInputArchive archive(is);

      archive(*this);

      std::cout << "Loaded vocabulary from " << filename << " with "
                << m_words.size() << " words." << std::endl;

    } else {
      std::cout << "Failed to load vocabulary " << filename << std::endl;
      std::abort();
    }
  }

 protected:
  /// Tree node
  struct Node {
    /// Node id
    NodeId id;
    /// Weight if the node is a word; may be positive or zero
    WordValue weight;
    /// Children
    std::vector<NodeId> children;
    /// Parent node (undefined in case of root)
    NodeId parent;
    /// Node descriptor
    TDescriptor descriptor;

    /// Word id if the node is a word
    WordId word_id;

    /**
     * Empty constructor
     */
    Node() : id(0), weight(0), parent(0), word_id(0) {}

    /**
     * Constructor
     * @param _id node id
     */
    Node(NodeId _id) : id(_id), weight(0), parent(0), word_id(0) {}

    /**
     * Returns whether the node is a leaf node
     * @return true iff the node is a leaf
     */
    inline bool isLeaf() const { return children.empty(); }

    template <class Archive>
    void serialize(Archive& ar) {
      ar(id, weight, children, parent, descriptor, word_id);
    }
  };

  template <class Archive>
  void save(Archive& ar) const {
    ar(CEREAL_NVP(this->m_k));
    ar(CEREAL_NVP(this->m_L));
    ar(CEREAL_NVP(this->m_nodes));
  }

  template <class Archive>
  void load(Archive& ar) {
    ar(CEREAL_NVP(this->m_k));
    ar(CEREAL_NVP(this->m_L));
    ar(CEREAL_NVP(this->m_nodes));

    createWords();
  }

  void createWords() {
    m_words.clear();

    if (!m_nodes.empty()) {
      m_words.reserve((int)pow((double)m_k, (double)m_L));

      for (Node& n : m_nodes) {
        if (n.isLeaf()) {
          n.word_id = m_words.size();
          m_words.push_back(&n);
        }
      }
    }
  }

  friend class cereal::access;

  /// Branching factor
  int m_k;

  /// Depth levels
  int m_L;

  /// Tree nodes
  std::vector<Node> m_nodes;

  /// Words of the vocabulary (tree leaves)
  /// this condition holds: m_words[wid]->word_id == wid
  std::vector<Node*> m_words;
};

}  // namespace visnav
