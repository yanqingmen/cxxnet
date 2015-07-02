#ifndef ITER_DOC_BATCH_INL_HPP
#define ITER_DOC_BATCH_INL_HPP
#pragma once

#include "data.h"
#include <map>
#include <algorithm>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cctype>

namespace cxxnet {
class DocBatchIterator : public IIterator<DataBatch> {
public:
  DocBatchIterator() {
    fi = NULL;
    fpdict = NULL;
    max_dict_size = 10000;
    t = 0;
    max_t = 0;
    batch_size = 10;
    top = 0;
  }
  virtual ~DocBatchIterator() {
    out.FreeSpaceDense();
  }
  virtual void SetParam(const char *name, const char *val) {
    if (!strcmp(name, "batch_size"))  batch_size = (index_t)atoi(val);
    if (!strcmp(name, "max_dict_size")) max_dict_size = atoi(val);
    if (!strcmp(name, "doc_file")) file_name = val;
    if (!strcmp(name, "dict_file")) dict_name = val;
  }
  virtual void Init(void) {
    dshape[0] = batch_size;
    dshape[1] = 1;
    dshape[2] = 1;
    dshape[3] = max_dict_size;
    out.AllocSpaceDense(dshape, batch_size, 1, false);
    loc.resize(batch_size, 0);
    this->LoadText();
  }
  virtual void BeforeFirst() {
    top = 0;
  }
  virtual bool Next() {
    out.data = 0.0f;
    out.label = 0.0f;
    for (index_t i = 0; i < batch_size; ++i) {
      size_t id = (i + top) % buf.size();
      if (loc[i] == buf[id].size()) {
        out.data[i][0][0][eof_idx] = 1.0f;
        out.label[i][0] = buf[id][0];
        loc[i] = 0;
      } else {
        out.data[i][0][0][buf[id][loc[i]]] = 1.0f;
        if (loc[i] + 1 == buf[id].size()) {
          out.label[i][0] = eof_idx;
        } else {
          out.label[i][0] = buf[id][loc[i] + 1];
        }
      }
      loc[i]++;
    }
    t += 1;
    if (top > buf.size()) return false;
    if (t >= max_t) {
      t = 0;
      top += batch_size;
      for (index_t i = 0; i< batch_size; ++i) {
        loc[i] = 0;
      }
      return false;
    }
    return true;
  }
  virtual const DataBatch &Value(void) const {
    return out;
  }
private:
  const static int word_size = 256;
  const static int unknown_idx = 0;
  const static int eof_idx = 1;
  struct DictRecord {
    char word[word_size];
    int idx;
  };
  struct cmp {
    bool operator() (const std::pair<std::string, int> &i, const std::pair<std::string, int> &j) {
      return (i.second > j.second);
    }
  } cmp;
  void LoadDict() {
    fpdict = utils::FopenCheck(dict_name.c_str(), "rb");
    DictRecord r;
    while (feof(fpdict)) {
      utils::Check(fread(&r, sizeof(DictRecord), 1, fpdict) > 0, "Incorrect dict file");
      dict[r.word] = r.idx;
    }
    fclose(fpdict);
    printf("Finishing loading dict\n");
  }
  void BuildDict() {
    size_t i = 0;
    std::map<std::string, int> cnt;
    char tok = '\n';
    std::string wd;
    if (fi != NULL) {
      fclose(fi);
    }
    fi = utils::FopenCheck(file_name.c_str(), "r");
    while (tok != EOF) {
      tok = fgetc(fi);
      if (ispunct(tok) || tok == ' ' || tok == '\t' || tok == '\n') {
        if (wd.size() > 0) {
          cnt[wd] += 1;
        }
        wd.clear();
      } else {
        wd += tok;
      }
    }
    fclose(fi);
    std::vector<std::pair<std::string, int> > tmp(cnt.size());
    for (std::map<std::string, int>::iterator it = cnt.begin(); it != cnt.end(); ++it) {
      tmp[i++] = std::make_pair(it->first, it->second);
    }
    std::sort(tmp.begin(), tmp.end(), cmp);
    const size_t max_num = std::min(max_dict_size - 2, tmp.size());
    for (i = 0; i < max_num; ++i) {
      dict[tmp[i].first] = i + 2;
    }
    DictRecord r;
    fpdict = utils::FopenCheck(dict_name.c_str(), "wb");
    for (i = 0; i < max_num; ++i) {
      utils::Check(tmp[i].first.size() < word_size, "error dict");
      strcpy(r.word, tmp[i].first.c_str());
      r.idx = i + 2;
      fwrite(&r, sizeof(DictRecord), 1, fpdict);
    }
    fclose(fpdict);
    printf("Finishing building dictionary\n");
  }
  void LoadText() {
    fpdict = fopen(dict_name.c_str(), "rb");
    if (fpdict == NULL) {
      printf("Can't find dict file, build a new one\n");
      this->BuildDict();
    } else {
      fclose(fpdict);
      this->LoadDict();
    }
    fi = utils::FopenCheck(file_name.c_str(), "r");
    char tok = '\n';
    std::string wd;
    std::vector<int> sentence;
    while (tok != EOF) {
      tok = fgetc(fi);
      if (tok == '\n') {
        if (wd.size() > 0) {
          if (dict.find(wd) == dict.end()) sentence.push_back(0); // unknown idx
          else sentence.push_back(dict[wd]);
        }
        if (sentence.size() > 0) {
          buf.push_back(sentence);
        }
        max_t = std::max(sentence.size(), max_t);
        wd.clear();
        sentence.clear();
      } else if (ispunct(tok) || tok == ' ' || tok == '\t') {
        if (wd.size() > 0) {
          if (dict.find(wd) == dict.end()) sentence.push_back(0); // unknow idx
          else sentence.push_back(dict[wd]);
          wd.clear();
        }
      } else {
        wd += tok;
      }
    }
    if (sentence.size() > 0) {
      buf.push_back(sentence);
    }
    printf("Finishing loading %lu sentences\n", buf.size());
  }
  DataBatch out;
  std::map<std::string, int > dict;
  std::vector<std::vector<int> > buf;
  std::vector<index_t> loc;
  std::string file_name;
  std::string dict_name;
  FILE *fi;
  FILE *fpdict;
  size_t max_t;
  size_t t;
  size_t max_dict_size;
  size_t top;
  index_t batch_size;
  mshadow::Shape<4> dshape;
  mshadow::Shape<2> lshape;
}; // class DocBatchIterator
}; // namespace cxxnet
#endif // ITER_DOC_BATCH_INL_HPP
