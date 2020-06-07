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
use darts::{DoubleArrayTrieBuilder, DoubleArrayTrie};

#[derive(Clone, Copy)]
pub struct Node<'a> {
    pub text: &'a str,
    pub score: f32,
    pub index: i32,
    pub start: usize,
    pub end: usize,
}

pub struct Prefix {
    pub text: String,
    pub len: usize,
    pub score: f32,
    pub index: i32,
}

pub struct SentencePieceModel {
    pub dart: DoubleArrayTrie,
    pub vocab: BrownHashMap<String, Prefix>,
}

impl SentencePieceModel {
    pub fn from_file(path: &str) -> SentencePieceModel {
        let mut f = File::open(path).unwrap();
        let mut contents = Vec::new();
        f.read_to_end(&mut contents).unwrap();

        let proto = parse_from_bytes::<ModelProto>(contents.as_slice()).unwrap();
        let mut vocab = BrownHashMap::new();
        let mut records: Vec<&str> = Vec::new();
        for (idx, piece) in proto.get_pieces().iter().enumerate() {
            let text = piece.get_piece();
            records.push(piece.get_piece());
            vocab.insert(text.to_owned(),
                Prefix {
                    text: text.to_owned(),
                    len: text.len(),
                    score: piece.get_score(),
                    index: idx as i32
                }
            );
        }
        records.sort_by(|a, b| a.cmp(&b));
        let dart = DoubleArrayTrieBuilder::new().build(&records);

        SentencePieceModel { dart, vocab }
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

    pub fn common_prefix_search<'a>(&'a self, text: &'a str) -> Vec<&Prefix> {
        self.dart.common_prefix_search(text).map(|matches| {
            matches
                .iter()
                .map(|(end_idx, _)| {
                    self.vocab.get(&text[..*end_idx]).unwrap()
                })
                .collect()
        }).unwrap_or(vec!())
    }

    pub fn decode_forward<'a>(&'a self, text: &'a str) -> Vec<Option<Node<'a>>> {
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

    pub fn tokenize_dag(&self, text: &str) -> Vec<String> {
        let text = text.replace(' ', "\u{2581}");
        let text = text.as_str();
        let output = self.decode_forward(text);
        let decoded = self.decode_backward(&output);
        decoded.into_iter().map(|node| node.text.to_string()).collect()
    }
}

fn main() {
    let text = " All human beings are born free and equal in dignity and rights. They are endowed with reason and conscience and should act towards one another in a spirit of brotherhood.";
    let tokenizer = SentencePieceModel::from_file("E:/Coding/notebooks/xlnet-base-cased-spiece.model");

    let now = Instant::now();
    for _ in 0..100 {
        let _ = tokenizer.tokenize_dag(text);

    }
    println!("{:?}", now.elapsed().as_nanos()/100);
    let output = tokenizer.tokenize_dag(text);
    println!("{:?}", output);
}