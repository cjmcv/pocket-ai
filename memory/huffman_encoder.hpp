#ifndef POCKET_AI_MEMORY_HUFFMAN_ENCODER_HPP_
#define POCKET_AI_MEMORY_HUFFMAN_ENCODER_HPP_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "util/logger.hpp"

namespace pai {
namespace memory {

#define MAX_SYMBOLS 256

// Get the ith bit in the bits array
static uint8_t get_bit(uint8_t* bits, uint64_t i) {
	return (bits[i / 8] >> i % 8) & 1;
}

class HuffmanEncoder {
typedef struct _Node {
	uint8_t is_leaf;
	uint64_t count;
	struct _Node *parent;
	struct _Node *zero, *one;
	uint8_t symbol;
} Node;

typedef struct _Code {
	_Code() {
		bits = nullptr;
		numbits = 0;
	}
	uint64_t numbits;
	// Binary encoding from root to leaf.
	uint8_t *bits;
} Code;

class Tree {
public:
	Tree() : root_(nullptr) {}

	Node *root() {
		if (root_ == nullptr)
			PAI_LOGE("root_ == nullptr.\n");
		return root_;
	}

	uint32_t GetSymbolFrequencies(const uint8_t *raw, uint32_t size) {

		memset(symbol_freqs_, 0, sizeof(Node *) * MAX_SYMBOLS);

		// Count the frequency of each symbol in the input data.
		uint32_t total_count = 0;
		for (uint32_t i = 0; i < size; ++i) {
			uint8_t c = raw[i];
			if (!symbol_freqs_[c])
				symbol_freqs_[c] = CreateLeafNode(c);
			++(symbol_freqs_[c]->count);
			++total_count;
		}

		return total_count;
	}

	// Build a huffman tree based on symbol frequency.
	void BuildTreeFromFreqs(bool is_debug) {

		// Sort the symbol frequency array by ascending frequency
		qsort(symbol_freqs_, MAX_SYMBOLS, sizeof(symbol_freqs_[0]), FreqsComp);

		if (is_debug) {
			printf("AFTER SORT\n");
			PrintFreqs();
		}

		// Skip the empty ones to get the number of symbols.
		uint32_t n = 0;
		for (n = 0; n < MAX_SYMBOLS && symbol_freqs_[n]; ++n)
			;

		for (uint32_t i = 1; i < n; ++i) {
			// Take the two with the least frequency.
			Node *n1 = symbol_freqs_[0];
			Node *n2 = symbol_freqs_[1];

			// Combine n1 and n2 to form a new non leaf node
			// and place it at the position of n1,
			// removing the entry point of n2.
			symbol_freqs_[0] = n1->parent = n2->parent = CreateNonLeafNode(n1->count + n2->count, n1, n2);
			symbol_freqs_[1] = NULL;

			// Reorder
			qsort(symbol_freqs_, n, sizeof(symbol_freqs_[0]), FreqsComp);
		}

		root_ = symbol_freqs_[0];
	}

	Node *CreateRoot() {
		root_ = CreateNonLeafNode(0, NULL, NULL);
		return root_;
	}

	void RebuildSubTree(Node *root, uint8_t *bytes, uint8_t numbits, uint8_t symbol) {
		// Add the entry to the Huffman tree.
		// The curbit is used switch between zero and one child nodes in the tree.
		// New nodes are added as needed.
		Node *p = root;
		for (uint32_t curbit = 0; curbit < numbits; ++curbit) {
			if (get_bit(bytes, curbit)) {
				if (p->one == NULL) {
					p->one = curbit == (uint8_t)(numbits - 1)
								 ? CreateLeafNode(symbol)
								 : CreateNonLeafNode(0, NULL, NULL);
					p->one->parent = p;
				}
				p = p->one;
			}
			else {
				if (p->zero == NULL) {
					p->zero = curbit == (uint8_t)(numbits - 1)
								  ? CreateLeafNode(symbol)
								  : CreateNonLeafNode(0, NULL, NULL);
					p->zero->parent = p;
				}
				p = p->zero;
			}
		}
	}

	void Release() {
		RecursiveRelease(root_);
	}

private:
	// Used for qsort default ascending order, 1: p2 p1, -1: p1 p2
	//  p1 > p2 => 1, p1 < p2 => -1, p1 == p2 => 0
	//  Putting the less frequent ones at the front and the more frequent ones at the back
	// allows for merging and building the Huffman tree from front to back,
	// resulting in longer links with fewer frequencies.
	static int32_t FreqsComp(const void *p1, const void *p2) {
		const Node *n1 = *(const Node **)p1;
		const Node *n2 = *(const Node **)p2;

		// Sort all NULLs to the end.
		if (n1 == NULL && n2 == NULL)
			return 0;
		if (n1 == NULL) // Return to 1, put the latter in front
			return 1;
		if (n2 == NULL) // Return to -1, put the former in front
			return -1;

		if (n1->count > n2->count)
			return 1;
		else if (n1->count < n2->count)
			return -1;

		return 0;
	}

	Node *CreateLeafNode(uint8_t symbol) {
		Node *p = (Node *)malloc(sizeof(Node));
		p->is_leaf = 1;
		p->symbol = symbol;
		p->count = 0;
		p->parent = 0;
		//
		p->zero = nullptr;
		p->one = nullptr;
		return p;
	}

	Node *CreateNonLeafNode(uint64_t count, Node *zero, Node *one) {
		Node *p = (Node *)malloc(sizeof(Node));
		p->is_leaf = 0;
		p->count = count;
		p->zero = zero;
		p->one = one;
		p->parent = 0;
		//
		p->symbol = 0;
		return p;
	}

	void PrintFreqs() {
		for (uint32_t i = 0; i < MAX_SYMBOLS; ++i) {
			if (symbol_freqs_[i])
				printf("(%d: %llu), ", symbol_freqs_[i]->symbol, symbol_freqs_[i]->count);
			else
				printf("NULL,");
		}
		printf("\n");
	}

	void RecursiveRelease(Node *subtree) {
		if (subtree == NULL)
			return;

		if (!subtree->is_leaf) {
			RecursiveRelease(subtree->zero);
			RecursiveRelease(subtree->one);
		}

		free(subtree);
	}

private:
	// Symbol frequencies
	Node *symbol_freqs_[MAX_SYMBOLS];
	// The root of huffman tree
	Node *root_;
};

class CodeTable {
public:
	CodeTable() {
		max_reverse_bytes_ = 2;
		reverse_tmp_ = (uint8_t *)malloc(max_reverse_bytes_);

		code_table_.resize(MAX_SYMBOLS);
	}

	void Release() {
		free(reverse_tmp_);

		for (uint32_t i = 0; i < code_table_.size(); ++i) {
			if (code_table_[i].bits != nullptr)
				free(code_table_[i].bits);
		}
	}

	inline std::vector<Code> &code_table() { return code_table_; }

	// Build code for each leaf node.
	void BuildCodeTable(Node *subtree) {
		if (subtree == NULL)
			return;

		// Builds a Code for each leaf in a huffman tree.
		if (subtree->is_leaf) {
			BuildCode(subtree, &code_table_[subtree->symbol]);
		}
		else {
			BuildCodeTable(subtree->zero);
			BuildCodeTable(subtree->one);
		}
	}

private:
	void ReverseBits(uint8_t *bits, uint64_t numbits) {
		uint64_t numbytes = (uint64_t)ceil((float_t)numbits / 8);

		if (max_reverse_bytes_ < numbytes) {
			max_reverse_bytes_ = numbytes;
			reverse_tmp_ = (uint8_t *)realloc(reverse_tmp_, max_reverse_bytes_);
		}
		memset(reverse_tmp_, 0, numbytes);

		long curbyte = 0;
		for (uint64_t curbit = 0; curbit < numbits; ++curbit) {
			uint32_t bitpos = curbit % 8;

			if (curbit > 0 && curbit % 8 == 0)
				++curbyte;

			reverse_tmp_[curbyte] |= (get_bit(bits, numbits - curbit - 1) << bitpos);
		}

		memcpy(bits, reverse_tmp_, numbytes);
	}

	// Generate encoding from leaf nodes to root nodes,
	// Then flip the encoding in order from the root node to the leaf node.
	// Convenient indexing from root to leaf nodes during encoding and decoding
	void BuildCode(const Node *leaf, Code *code) {

		uint64_t numbits = 0;
		uint8_t *bits = NULL;

		while (leaf && leaf->parent) {
			Node *parent = leaf->parent;
			uint8_t cur_bit = (uint8_t)(numbits % 8);
			uint64_t cur_byte = numbits / 8;

			// If we need another byte to hold the code, then allocate it.
			if (cur_bit == 0) {
				size_t newSize = cur_byte + 1;
				bits = (uint8_t *)realloc(bits, newSize);
				bits[newSize - 1] = 0;
			}

			// If leaf is a One, use OR operation to add 1 to the corresponding position.
			// If leaf is a Zero, then do nothing, since the bits was initialized to zero.
			if (leaf == parent->one)
				bits[cur_byte] |= 1 << cur_bit;

			++numbits;
			// Walk towards the root node direction
			leaf = parent;
		}

		if (bits)
			ReverseBits(bits, numbits);

		code->numbits = numbits;
		code->bits = bits;
	}

private:
	std::vector<Code> code_table_;
	uint8_t *reverse_tmp_;
	uint64_t max_reverse_bytes_;
};

public:
	HuffmanEncoder() {
		encoded_data_ = nullptr;
		encoded_size_ = 0;
		decoded_data_ = nullptr;
		decoded_size_ = 0;
	};

	int32_t Encode(const uint8_t *bufin, uint32_t bufin_size,
				uint8_t **bufout, uint32_t *bufout_size) {

		if (!bufout || !bufout_size)
			return 1;

		// Get the frequency of each symbol in the input memory.
		Tree tree;
		uint32_t symbol_count = tree.GetSymbolFrequencies(bufin, bufin_size);
		tree.BuildTreeFromFreqs(false);

		// an array of huffman code index by symbol value.
		CodeTable table;
		table.BuildCodeTable(tree.root());

		if (encoded_data_ != nullptr)
			free(encoded_data_);
		encoded_data_ = (uint8_t *)malloc(bufin_size);

		WriteCodeTable(table.code_table(), symbol_count);
		WriteCode(bufin, bufin_size, table.code_table());

		tree.Release();
		table.Release();

		encoded_data_ = (uint8_t *)realloc(encoded_data_, encoded_size_);
		*bufout = encoded_data_;
		*bufout_size = encoded_size_;
		return 0;
	}

	int32_t Decode(const uint8_t *bufin, uint32_t bufin_size,
				uint8_t **bufout, uint32_t *bufout_size) {

		if (!bufout || !bufout_size)
			return 1;

		// Read the huffman code table from memory and rebuild the huffman tree.
		Tree tree;
		uint32_t data_count;
		uint32_t index = 0;
		ReadCodeTableRebuildTree(bufin, bufin_size, &tree, &index, &data_count);

		if (decoded_data_ != nullptr)
			free(decoded_data_);
		decoded_data_ = (uint8_t *)malloc(data_count);
		decoded_size_ = 0;

		// Skip the code table and decode the reset.
		ReadCode(bufin, bufin_size, index, tree.root(), data_count);

		tree.Release();

		*bufout = decoded_data_;
		*bufout_size = decoded_size_;
		return 0;
	}

	void Release() {
		if (encoded_data_ != nullptr) {
			free(encoded_data_);
			encoded_data_ = nullptr;
			encoded_size_ = 0;
		}
		if (decoded_data_ != nullptr) {
			free(decoded_data_);
			decoded_data_ = nullptr;
			decoded_size_ = 0;
		}
	}

private:
	void Write(const void *to_write, uint32_t to_write_len) {
		uint32_t newlen = encoded_size_ + to_write_len;
		memcpy(encoded_data_ + encoded_size_, to_write, to_write_len);
		encoded_size_ = newlen;
	}

	// Write code table to the output buffer.
	void WriteCodeTable(std::vector<Code> &tbl, uint32_t symbol_count) {

		// Get the number of entries in code table.
		uint32_t count = 0;
		for (uint32_t i = 0; i < MAX_SYMBOLS; ++i) {
			if (tbl[i].bits != nullptr)
				++count;
		}
		Write(&count, sizeof(count));

		// Write the number of bytes that will be encoded.
		Write(&symbol_count, sizeof(symbol_count));

		// Write the entries.
		for (uint32_t i = 0; i < MAX_SYMBOLS; ++i) {
			if (tbl[i].bits != nullptr) {

				uint8_t uc = (uint8_t)i;
				// Write the 1 byte symbol.
				Write(&uc, sizeof(uc));
				// Write the 1 byte code bit length.
				uc = (uint8_t)tbl[i].numbits;
				Write(&uc, sizeof(uc));
				// Write the code bytes.
				uint32_t numbytes = (uint32_t)ceil((float_t)tbl[i].numbits / 8);
				Write(tbl[i].bits, numbytes);
			}
		}
	}

	void ReadCodeTableRebuildTree(const uint8_t *bufin,
								uint32_t bufin_size,
								Tree *tree,
								uint32_t *cur_index,
								uint32_t *data_bytes) {

		uint8_t *cur_ptr = (uint8_t *)bufin;

		// Read the number of entries.
		uint32_t count;
		memcpy(&count, cur_ptr, sizeof(count));
		cur_ptr += sizeof(count);

		// Read the number of bytes that have been encoded.
		memcpy(data_bytes, cur_ptr, sizeof(*data_bytes));
		cur_ptr += sizeof(*data_bytes);

		Node *root = tree->CreateRoot();
		/* Read the entries. */
		while (count-- > 0) {

			// Read the 1 byte symbol.
			uint8_t symbol;
			memcpy(&symbol, cur_ptr, sizeof(symbol));
			cur_ptr += sizeof(symbol);

			// Read the 1 byte code bit length.
			uint8_t numbits;
			memcpy(&numbits, cur_ptr, sizeof(numbits));
			cur_ptr += sizeof(numbits);

			// Read the code bytes.
			uint8_t numbytes = (uint8_t)ceil((float_t)numbits / 8);
			uint8_t *bytes = (uint8_t *)malloc(numbytes);
			memcpy(bytes, cur_ptr, numbytes);
			cur_ptr += numbytes;

			// Rebuild tree.
			tree->RebuildSubTree(root, bytes, numbits, symbol);

			free(bytes);
		}

		*cur_index = cur_ptr - bufin;
	}

	void WriteCode(const uint8_t *bufin,
				uint32_t bufin_size,
				std::vector<Code> &tbl) {
		uint8_t curbyte = 0;
		uint8_t curbit = 0;

		for (uint32_t i = 0; i < bufin_size; ++i) {
			uint8_t uc = bufin[i];
			for (uint64_t j = 0; j < tbl[uc].numbits; ++j) {
				// Add the current bit to curbyte.
				curbyte |= get_bit(tbl[uc].bits, j) << curbit;

				// Write only when there are enough 8 bits
				if (++curbit == 8) {
					Write(&curbyte, sizeof(curbyte));
					curbyte = 0;
					curbit = 0;
				}
			}
		}

		// Write the remaining content that has not reached 8 bits
		if (curbit > 0)
			Write(&curbyte, sizeof(curbyte));
	}

	void ReadCode(const uint8_t *bufin, uint32_t bufin_size, uint32_t code_offset, Node *root, uint32_t data_size) {
		// Skip the code table and decode the reset.
		Node *p = root;
		for (uint32_t i = code_offset; i < bufin_size && data_size > 0; ++i) {
			uint8_t byte = bufin[i];
			uint8_t mask = 1;
			while (data_size > 0 && mask) {
				p = byte & mask ? p->one : p->zero;
				mask <<= 1;

				if (p->is_leaf) {
					decoded_data_[decoded_size_++] = p->symbol;
					p = root;
					--data_size;
				}
			}
		}
	}

private:
	uint32_t encoded_size_;
	uint8_t *encoded_data_;

	uint32_t decoded_size_;
	uint8_t *decoded_data_;
};

} // memory.
} // pai.

#endif // POCKET_AI_MEMORY_HUFFMAN_ENCODER_HPP_
