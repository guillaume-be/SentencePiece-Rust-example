// Copyright 2019 Google LLC. All Rights Reserved.
// Copyright 2019-2020 Guillaume Becquin
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use hashbrown::HashMap as BrownHashMap;
use std::fs::File;
use protobuf::parse_from_bytes;
use rust_tokenizers::preprocessing::vocab::sentencepiece_proto::sentencepiece_model::ModelProto;
use std::io::Read;
use itertools::Itertools;
use std::time::Instant;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct Node<'a> {
    pub text: &'a str,
    pub score: f32,
    pub index: i32,
    pub start: usize,
    pub end: usize,
}

pub struct DagNode {
    pub text: String,
    pub len: usize,
    pub score: f32,
    pub index: i32,
    pub leaf: bool,
    pub children: BrownHashMap<char, DagNode>,
}

impl DagNode {
    pub fn new(text: String) -> DagNode {
        let len = text.chars().count();
        DagNode {
            text,
            len,
            score: 0.0,
            index: 0,
            leaf: false,
            children: BrownHashMap::new(),
        }
    }
}

pub struct SentencePieceModel {
    pub vocab: HashMap<String, (f32, i32)>,
    pub root: DagNode
}

impl SentencePieceModel {
    pub fn from_file(path: &str) -> SentencePieceModel {
        let mut f = File::open(path).unwrap();
        let mut contents = Vec::new();
        f.read_to_end(&mut contents).unwrap();

        let proto = parse_from_bytes::<ModelProto>(contents.as_slice()).unwrap();
        let vocab = HashMap::new();
        let root = DagNode::new("".to_string());
        let mut model = SentencePieceModel { root, vocab };
        for (idx, piece) in proto.get_pieces().iter().enumerate() {
            model.insert(piece.get_piece(), piece.get_score(), idx as i32);
            model.vocab.insert(piece.get_piece().to_owned(), (piece.get_score(), idx as i32));
        }
        model

    }

    fn insert(&mut self, word: &str, score: f32, index: i32) {
        let char_count = word.chars().count();
        let mut node = &mut self.root;

        for (idx, character) in word.chars().enumerate() {
            if !node.children.contains_key(&character) {
                let mut text = node.text.clone();
                text.push(character);
                let new_node = DagNode::new(text);
                node.children.insert(character, new_node);
            }
            node = node.children.get_mut(&character).unwrap();
            if idx == char_count - 1 {
                node.leaf = true;
                node.score = score;
                node.index = index;
            }
        }
    }

    pub fn decode_forward<'a>(&'a self, text: &'a str) -> Vec<Option<Node>> {
        let mut char_positions = text
            .char_indices()
            .map(|(pos, _)| pos)
            .collect_vec();
        char_positions.push(text.len());
        let mut results = vec!(None; char_positions.len());
        let mut scores = vec!(std::f32::NEG_INFINITY; char_positions.len());
        scores[0] = 0f32;

        for char_end in 0..char_positions.len() {
            for char_start in 0..char_end {
                let sub_text = &text[char_positions[char_start]..char_positions[char_end]];
                if let Some(subtoken) = self.vocab.get(sub_text) {
                    let local_score = scores[char_start] + subtoken.0;
                    if local_score > scores[char_end] {
                        results[char_end] = Some(Node {
                            text: sub_text,
                            score: local_score,
                            index: subtoken.1,
                            start: char_start,
                            end: char_end,
                        });
                        scores[char_end] = local_score;
                    }
                }
            }
            if scores[char_end] <= std::f32::MIN {
                results[char_end] = Some(Node {
                    text: &text[char_positions[char_end - 1]..char_positions[char_end]],
                    score: std::f32::MIN,
                    index: 0,
                    start: char_end - 1,
                    end: char_end,
                });
                scores[char_end] = 0f32;
            }
        }
        results
    }

    pub fn decode_backward<'a>(&'a self, nodes: &'a Vec<Option<Node<'a>>>) -> Vec<&'a Node> {
        let mut next_node = nodes.last().unwrap();
        let mut best_sequence = vec!();

        while next_node.is_some() {
            let node_value = next_node.as_ref().unwrap();
            best_sequence.push(node_value);
            next_node = &nodes[node_value.start];
        };
        best_sequence.reverse();
        best_sequence
    }

    pub fn common_prefix_search<'a>(&'a self, text: &'a str) -> Vec<&DagNode> {
        let mut results = vec!();
        let mut characters = text.chars();

        let mut node = self.root.children.get(&characters.next().unwrap());
        if node.is_some() {
            if node.unwrap().leaf {
                results.push(node.unwrap());
            }
        } else {
            return vec!();
        }
        while let Some(character) = characters.next() {
            node = node.unwrap().children.get(&character);
            if node.is_some() {
                if node.unwrap().leaf {
                    results.push(node.unwrap());
                }
            } else {
                break;
            }
        }

        results
    }

    pub fn decode_forward_dag<'a>(&'a self, text: &'a str) -> Vec<Option<Node<'a>>> {
        let mut char_positions = text
            .char_indices()
            .map(|(pos, _)| pos)
            .collect_vec();
        char_positions.push(text.len());
        let mut results = vec!(None; char_positions.len());
        let mut scores = vec!(std::f32::NEG_INFINITY; char_positions.len());
        scores[0] = 0f32;

        for char_start in 0..char_positions.len() - 1 {
            let matches = self.common_prefix_search(&text[char_positions[char_start]..]);
            for node in matches {
                let local_score = scores[char_start] + node.score;
                let char_end = char_start + node.len;
                if local_score > scores[char_end] {
                    results[char_end] = Some(Node {
                        text: &text[char_positions[char_start]..char_positions[char_end]],
                        score: local_score,
                        index: node.index,
                        start: char_start,
                        end: char_end,
                    });
                    scores[char_end] = local_score;
                }
            }
            if scores[char_start + 1] <= std::f32::MIN {
                results[char_start + 1] = Some(Node {
                    text: &text[char_positions[char_start]..char_positions[char_start + 1]],
                    score: std::f32::MIN,
                    index: 0,
                    start: char_start,
                    end: char_start + 1,
                });
                scores[char_start + 1] = 0f32;
            }
        }
        results
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let text = text.replace(' ', "\u{2581}");
        let text = text.as_str();
        let output = self.decode_forward(text);
        let decoded = self.decode_backward(&output);
        decoded.into_iter().map(|node| node.text.to_string()).collect()
    }

    pub fn tokenize_dag(&self, text: &str) -> Vec<String> {
        let text = text.replace(' ', "\u{2581}");
        let text = text.as_str();
        let output = self.decode_forward_dag(text);
        let decoded = self.decode_backward(&output);
        decoded.into_iter().map(|node| node.text.to_string()).collect()
    }
}

fn main() {

let text = " The quick brown fox jumps over the lazy dog";

    let tokenizer = SentencePieceModel::from_file("path/to/spiece.model");

    let now = Instant::now();
    for _ in 0..100 {
        let _ = tokenizer.tokenize_dag(text);
    }

    println!("{:?}", now.elapsed().as_nanos()/100);

}