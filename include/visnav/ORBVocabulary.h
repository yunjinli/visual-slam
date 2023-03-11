#ifndef ORBVOCABULARY_H
#define ORBVOCABULARY_H

// #include <DBoW2/TemplatedVocabulary.h>
// #include <visnav/FORB.h>

#include <DBoW2/FORB.h>
#include <DBoW2/TemplatedVocabulary.h>

namespace visnav {

// typedef DBoW2::TemplatedVocabulary<FORB::TDescriptor, FORB> ORBVocabulary;
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
    ORBVocabulary;

}  // namespace visnav

#endif  // ORBVOCABULARY_H