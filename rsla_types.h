//
// Created by chlorinepentoxide on 7/1/25.
//

#ifndef RSLA_TYPES_H
#define RSLA_TYPES_H

#ifndef USE_ARMA
#ifndef USE_MPLAPACK
#define USE_MPLAPACK
#endif
#endif

#include <valarray>
#include <string>

#ifdef USE_MPLAPACK
#include <mplapack_config.h>
#include "mpreal.h"
#endif

namespace rsla {

#ifdef USE_ARMA
    typedef double       real;
    typedef int          integer;
    typedef unsigned int uinteger;

    const std::string REAL_TYPE = "DOUBLE64";
    const std::string INT_TYPE  = "INT32";
    const std::string UINT_TYPE = "UINT32";
#endif

#ifdef USE_MPLAPACK
    typedef mpfr::mpreal       real;
    typedef signed long int    integer;
    typedef  long int  uinteger;

    const std::string REAL_TYPE = "MPFR/MPREAL";
    const std::string INT_TYPE  = "INT64";
    const std::string UINT_TYPE = "UINT64";
#endif

    typedef std::valarray<uinteger>      integer_array;
    typedef std::valarray<real>          real_array;
    typedef std::valarray<integer_array> integer_table;
    typedef std::valarray<real_array>    real_table;
    typedef std::valarray<integer_table> integer_tables;
    typedef std::valarray<real_table>    real_tables;

    inline std::string to_string (const std::valarray<uinteger>& array) {
        std::string val = "[";
        for(uinteger i = 0; i < array.size(); i++) {
            val += std::to_string(array[i]);
            if(i != array.size() - 1)
                val += ", ";
        }
        return val + "]";
    }

    inline uinteger max (const std::valarray<uinteger>& array) {
        uinteger val = array[0];
        for(uinteger i = 0; i < array.size(); i++) if(array[i] > val) val = array[i];
        return val;
    }

    inline uinteger sum (const std::valarray<uinteger>& array) {
        uinteger val = 0;
        for(uinteger i = 0; i < array.size(); i++) val += array[i];
        return val;
    }

}

#endif //RSLA_TYPES_H
