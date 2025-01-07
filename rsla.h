//
// Created by chlorinepentoxide on 7/1/25.
//

#ifndef RSLA_H
#define RSLA_H

#include <iostream>
#include <cmath>

#include "rsla_types.h"

namespace rsla {

    const std::string RSLA_VERSION = "TITAN 240";

#ifdef USE_ARMA
    const std::string RSLA_BACKEND = "Armadillo LAPACK Wrappers version 1.0.0";
#endif
#ifdef USE_MPLAPACK
    const std::string RSLA_BACKEND = "RSLA MPLAPACK Wrappers version 1.0.0";
#endif

    class RSVector {
    private:
        real* data;
        uinteger len;
    public:
        RSVector() {
            data = new real[1];
            len = 1;
        }

        ~RSVector() {
            delete[] data;
        }

        RSVector(const RSVector& v) {
            data = new real [v.len];
            std::copy (v.data, v.data + v.len, data);
            len = v.len;
        }

        explicit RSVector(const uinteger sz) {
            data = new real[sz];
            for(uinteger i = 0; i < sz; i++) data[i] = 0;
            len = sz;
        }

        RSVector(real* dat, const uinteger sz) {
            data = new real [sz];
            std::copy(dat, dat + sz, data);
            len = sz;
        }

        RSVector& operator = (const RSVector& other) {
            if(this != &other) {
                real* copy = new real [other.len];
                std::copy (other.data, other.data + len, copy);
                delete[] data;
                data = copy;
                len = other.len;
            }
            return *this;
        }

        RSVector operator + (const RSVector& a) const {
            if(len != a.len) {
                std::cerr << "[ERROR] RSVector Add Operation -- Incompatible Sizes (" << len << ", " << a.len << ") \n";
                exit(100);
            }
            real* dat = new real[len];
            for(uinteger i = 0; i < len; i++) dat[i] = data[i] + a.data[i];
            RSVector result (dat, len);
            delete[] dat;
            return result;
        }

        RSVector operator - (const RSVector& a) const {
            if(len != a.len) {
                std::cerr << "[ERROR] RSVector Sub Operation -- Incompatible Sizes (" << len << ", " << a.len << ") \n";
                exit(100);
            }
            real* dat = new real[len];
            for(uinteger i = 0; i < len; i++) dat[i] = data[i] - a.data[i];
            RSVector result (dat, len);
            delete[] dat;
            return result;
        }

        RSVector operator * (const real& a) const {
            real* dat = new real[len];
            for(uinteger i = 0; i < len; i++) dat[i] = data[i] * a;
            RSVector result (dat, len);
            delete[] dat;
            return result;
        }

        real operator * (const RSVector& a) const {
            if(len != a.len) {
                std::cerr << "[ERROR] RSVector Add Operation -- Incompatible Sizes (" << len << ", " << a.len << ") \n";
                exit(100);
            }
            real sum = 0;
            for(uinteger i = 0; i < len; i++) sum  += data[i] * a.data[i];
            return sum;
        }

        void set(const uinteger index, const real& value) const {
            if(index >= len) {
                std::cerr << "[ERROR] RSVector Set Operation -- Illegal Access (" << index << ", " << len << ") \n";
                exit(101);
            }
            data[index] = value;
        }

        [[nodiscard]] real get(const uinteger index) const {
            if(index >= len) {
                std::cerr << "[ERROR] RSVector Get Operation -- Illegal Access (" << index << ", " << len << ") \n";
                exit(101);
            }
            return data[index];
        }

        void addto (const uinteger i, const real& value) {
            set(i, get(i) + value);
        }

        [[nodiscard]] uinteger size() const {
            return len;
        }

        [[nodiscard]] real l2norm() const {
            real norm = 0;
            for(uinteger i = 0; i < len; i++) norm += (pow(data[i], 2) / real(len));
            return sqrt(norm);
        }

        real** getmemptr() {
            return &data;
        }

        void sort () const {
            std::sort(data, data + len);
        }

        void print() const {
            std::cout << "\n";
            for(uinteger i = 0; i < len; i++) std::cout << data[i] << "\n";
            std::cout << "\n";
        }
    };

    class RSMatrix {
    private:
        real* data;
        uinteger lrows;
        uinteger lcols;
    public:
        RSMatrix() {
            data = new real[1];
            lrows = 1;
            lcols = 1;
        }

        ~RSMatrix() {
            delete[] data;
        }

        RSMatrix(const RSMatrix& m) {
            data = new real [m.lrows * m.lcols];
            std::copy(m.data, m.data + m.lrows * m.lcols, data);
            lrows = m.lrows;
            lcols = m.lcols;
        }

        RSMatrix& operator = (const RSMatrix& other) {
            if(this != &other) {
                real* copy = new real [other.lrows * other.lcols];
                std::copy(other.data, other.data + other.lrows * other.lcols, copy);
                delete[] data;
                data = copy;
                lrows = other.lrows;
                lcols = other.lcols;
            }
            return *this;
        }

        explicit RSMatrix(real* dat, const uinteger r, const uinteger c) {
            real* cpy = new real [r * c];
            std::copy(dat, dat + r * c, cpy);
            data = cpy;
            lrows = r;
            lcols = c;
        }

        RSMatrix(const uinteger r, const uinteger c) {
            data = new real[r * c];
            for(uinteger i = 0; i < r * c; i++) data[i] = 0;
            lrows = r;
            lcols = c;
        }

        RSMatrix operator + (const RSMatrix& A) const {
            if(lcols != A.lcols || lrows != A.lrows) {
                std::cerr << "[ERROR] RSMatrix Add Operation -- Incompatible Shapes (" << lrows << ", " << lcols << ") (" << A.lrows << ", " << A.lcols << ") \n";
                exit(200);
            }
            real* dat = new real[lrows * lcols];
            for(uinteger i = 0; i < lrows * lcols; i++) dat[i] = data[i] + A.data[i];
            RSMatrix result (dat, lrows, lcols);
            delete[] dat;
            return result;
        }

        RSMatrix operator - (const RSMatrix& A) const {
            if(lcols != A.lcols || lrows != A.lrows) {
                std::cerr << "[ERROR] RSMatrix Sub Operation -- Incompatible Shapes (" << lrows << ", " << lcols << ") (" << A.lrows << ", " << A.lcols << ") \n";
                exit(200);
            }
            real* dat = new real[lrows * lcols];
            for(uinteger i = 0; i < lrows * lcols; i++) dat[i] = data[i] - A.data[i];
            return RSMatrix (dat, lrows, lcols);
        }

        RSMatrix operator * (const real& A) const {
            real* dat = new real[lrows * lcols];
            for(uinteger i = 0; i < lrows * lcols; i++) dat[i] = data[i] * A;
            RSMatrix result (dat, lrows, lcols);
            delete[] dat;
            return result;
        }

        [[nodiscard]] uinteger rows() const {
            return lrows;
        }

        [[nodiscard]] uinteger cols() const {
            return lcols;
        }

        real** getmemptr() {
            return &data;
        }

        void set(const uinteger i, const uinteger j, const real& value) const {
            if(lcols <= j || lrows <= i) {
                std::cerr << "[ERROR] RSMatrix Set Operation -- Illegal Access (" << i << ", " << j << ") (" << lrows << ", " << lcols << ") \n";
                exit(200);
            }
            data[j * lrows + i] = value;
        }

        [[nodiscard]] real get(const uinteger i, const uinteger j) const {
            if(lcols <= j || lrows <= i) {
                std::cerr << "[ERROR] RSMatrix Set Operation -- Illegal Access (" << i << ", " << j << ") (" << lrows << ", " << lcols << ") \n";
                exit(200);
            }
            return data[j * lrows + i];
        }

        void addto (const uinteger i, const uinteger j, const real& value) const {
            set(i, j, get(i, j) + value);
        }

        void make_identity() const {
            const uinteger m = std::min(lrows, lcols);
            for(uinteger i = 0; i < m; i++)
                set(i, i, 1);
        }

        void print() const {
            std::cout << "\n";
            for(uinteger i = 0; i < lrows; i++) {
                for(uinteger j = 0; j < lcols; j++)
                    std::cout << get(i, j) << " ";
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    };

    class RSSBMatrix {
    private:
        uinteger lrows;
        uinteger lcols;
        bool* tbands;
        bool* rbands;
        real* data;

        RSSBMatrix(const uinteger r, const uinteger c, bool* tb, bool* rb) {
            lrows = r;
            lcols = c;
            tbands = new bool [c];     std::copy(tb, tb + c, tbands);
            rbands = new bool [r - 1]; std::copy(rb, rb + r - 1, rbands);
            data = new real [r * c];
            for(uinteger i = 0; i < r * c; i++) data[i] = 0;
        }
    public:
        RSSBMatrix() {
            lrows = 1;
            lcols = 1;
            tbands = new bool [1] { false };
            rbands = new bool [1] { false };
            data = new real [1];
            for(uinteger i = 0; i < 1; i++) data[i] = 0;
        }

        RSSBMatrix(const uinteger r, const uinteger c) {
            lrows = r;
            lcols = c;
            tbands = new bool [c];
            rbands = new bool [r - 1];
            data = new real [r * c];
            for(uinteger i = 0; i < r * c; i++) data[i] = 0;
        }

        ~RSSBMatrix() {
            delete[] data;
            delete[] tbands;
            delete[] rbands;
        }

        RSSBMatrix (const RSSBMatrix& m) {
            lrows = m.lrows;
            lcols = m.lcols;
            tbands = new bool [m.lcols]; std::copy(m.tbands, m.tbands + m.lcols, tbands);
            rbands = new bool [m.lrows - 1]; std::copy(m.rbands, m.rbands + m.lrows - 1, rbands);
            data = new real [lrows * lcols]; std::copy(m.data, m.data + lrows * lcols, data);
        }

        RSSBMatrix& operator = (const RSSBMatrix& other) {
            if(this != &other) {
                lrows = other.lrows;
                lcols = other.lcols;
                tbands = new bool [lcols];
                rbands = new bool [lrows - 1];
                data = new real [lrows * lcols];
                std::copy(other.tbands, other.tbands + lcols, tbands);
                std::copy(other.rbands, other.rbands + lrows - 1, rbands);
                std::copy(other.data, other.data + lrows * lcols, data);
            }
            return *this;
        }

        RSSBMatrix operator + (const RSSBMatrix& A) const {
            if(lcols != A.lcols || lrows != A.lrows) {
                std::cerr << "[ERROR] RSSBMatrix Add Operation -- Incompatible Shapes (" << lrows << ", " << lcols << ") (" << A.lrows << ", " << A.lcols << ") \n";
                exit(300);
            }
            RSSBMatrix C (*this);
            for(uinteger i = 0; i < lcols; i++)
                if(A.tbands[i])
                    for(uinteger j = 0, k = i; k < lcols; j++, k++)
                        C.set(j, k, C.get(j, k) + A.get(j, k));
            for(uinteger i = 0; i < lrows; i++)
                if(A.rbands[i])
                    for(uinteger j = 0, k = i; k < lcols; j++, k++)
                        C.set(j, k, C.get(j, k) + A.get(j, k));
            return C;
        }

        RSSBMatrix operator - (const RSSBMatrix& A) const {
            if(lcols != A.lcols || lrows != A.lrows) {
                std::cerr << "[ERROR] RSSBMatrix Sub Operation -- Incompatible Shapes (" << lrows << ", " << lcols << ") (" << A.lrows << ", " << A.lcols << ") \n";
                exit(300);
            }
            RSSBMatrix C (*this);
            for(uinteger i = 0; i < lcols; i++)
                if(A.tbands[i])
                    for(uinteger j = 0, k = i; k < lcols; j++, k++)
                        C.set(j, k, C.get(j, k) - A.get(j, k));
            for(uinteger i = 0; i < lrows; i++)
                if(A.rbands[i])
                    for(uinteger j = 0, k = i; k < lcols; j++, k++)
                        C.set(j, k, C.get(j, k) - A.get(j, k));
            return C;
        }

        RSSBMatrix operator - (const real& A) const {
            RSSBMatrix C (*this);
            for(uinteger i = 0; i < lcols; i++)
                if(C.tbands[i])
                    for(uinteger j = 0, k = i; k < lcols; j++, k++)
                        C.set(j, k, C.get(j, k) * A);
            for(uinteger i = 0; i < lrows; i++)
                if(C.rbands[i])
                    for(uinteger j = 0, k = i; k < lcols; j++, k++)
                        C.set(j, k, C.get(j, k) * A);
            return C;
        }

        RSSBMatrix operator * (const RSSBMatrix& A) const {
            if(lcols != A.lrows) {
                std::cerr << "[ERROR] RSSBMatrix Mul Operation -- Incompatible Shapes (" << lrows << ", " << lcols << ") (" << A.lrows << ", " << A.lcols << ") \n";
                exit(300);
            }
            RSSBMatrix res (lrows, A.lcols);
            // Upper
            for(uinteger i = 0; i < A.lcols; i++)
                if(A.tbands[i])
                    for(uinteger j = 0, k = i; k < A.lcols; j++, k++)
                        for(uinteger l = 0; l < lrows; l++)
                            if(get(l, j) != 0.0)
                                res.set(l, k, res.get(l, k) + get(l, j) * A.get(j, k));
            // Lower
            for(uinteger i = 0; i < A.lrows - 1; i++)
                if(A.rbands[i])
                    for(uinteger j = i + 1, k = 0; j < A.lrows; j++, k++)
                        for(uinteger l = 0; l < lrows; l++)
                            if(get(l, j) != 0.0)
                                res.set(l, k, res.get(l, k) + get(l, j) * A.get(j, k));
            return res;
        }

        void set(const uinteger i, const uinteger j, const real& value) const {
            if(lcols <= j || lrows <= i) {
                std::cerr << "[ERROR] RSSBMatrix Set Operation -- Illegal Access (" << i << ", " << j << ") (" << lrows << ", " << lcols << ") \n";
                exit(300);
            }
            if(i <= j)
                tbands[j - i] = true;
            else
                rbands[i - j - 1] = true;
            data[j * lrows + i] = value;
        }

        [[nodiscard]] real get(const uinteger i, const uinteger j) const {
            if(lcols <= j || lrows <= i) {
                std::cerr << "[ERROR] RSSBMatrix Set Operation -- Illegal Access (" << i << ", " << j << ") (" << lrows << ", " << lcols << ") \n";
                exit(300);
            }
            return data[j * lrows + i];
        }

        [[nodiscard]] uinteger rows() const {
            return lrows;
        }

        [[nodiscard]] uinteger cols() const {
            return lcols;
        }

        real** getmemptr () {
            return &data;
        }

        void print() const {
            std::cout << "\n";
            for(uinteger i = 0; i < lrows; i++) {
                for(uinteger j = 0; j < lcols; j++)
                    std::cout << get(i, j) << " ";
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    };

    struct RSEigenResult {
        RSVector eigenvalues_real;
        RSVector eigenvalues_imag;
        RSMatrix eigenvector_real;
        RSMatrix eigenvector_imag;
    };

    // Implementation Defined Functions

    RSMatrix multiply (RSMatrix&, RSMatrix&);
    RSVector multiply (RSMatrix&, RSVector&);

    real condition_number (RSMatrix&);

    RSEigenResult eigen (RSMatrix&);
    RSEigenResult eigen (RSMatrix&, RSMatrix&);

    RSMatrix inv (RSMatrix&);
    RSMatrix pinv (RSMatrix&);

    RSVector solve (RSMatrix&, RSVector&);
    RSVector fast_solve (RSMatrix&, RSVector&);

}

#endif //RSLA_H
