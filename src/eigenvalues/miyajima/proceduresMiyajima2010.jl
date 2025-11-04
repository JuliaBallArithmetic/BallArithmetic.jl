# procedures from Miyajima2010, Section 4

function _encR1ci(A, tildeX, tildeD, Y, Zrl, Zru, Zil, Ziu)
    Crl, Cru, Cil, Ciu = _cprod(A, tildeX)
    Vrl, Vru, Vil, Viu = _ciprod_prime(Zrl, Zru, Zil, Ziu, tildeD)
    Url, Uil = setrounding(T, RoundDown) do
        Url = Crl - Vru
        Uil = Cil - Viu
        return Url, Uil
    end
    Uru, Uiu = setrounding(T, RoundUp) do
        Uru = Cru - Vrl
        Uiu = Ciu - Vil
        return Uru, Uiu
    end
    return _ciprod(Y, Url, Uru, Uil, Uiu)
end

function _encR1ccr(A, tildeX, tildeD, Y, Zc, Zr)
    Crl, Cru, Cil, Ciu = _cprod(A, tildeX)
    Vrl, Vru, Vil, Viu = _ccrprod_prime(Zc, Zr, tildeD)
    Url, Uil = setrounding(T, RoundDown) do
        Url = Crl - Vru
        Uil = Cil - Viu
        return Url, Uil
    end
    Uru, Uiu = setrounding(T, RoundUp) do
        Uru = Cru - Vrl
        Uiu = Ciu - Vil
        return Uru, Uiu
    end
    Uc, Ur = _ccr(Url, Uru, Uil, Uiu)
    Wrl, Wru, Wil, Wiu = _ccrprod(Y, Uc, Ur)
    return Wrl, Wru, Wil, Wiu
end
