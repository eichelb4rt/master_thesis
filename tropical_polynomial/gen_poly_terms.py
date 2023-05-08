REPLACE_A_PLUS_AND_A_MINUS = True


def next_h(last_f: str, last_g: str, layer: int) -> str:
    if REPLACE_A_PLUS_AND_A_MINUS:
        return f"\max(A^{{({layer})}}, 0){last_f} + (-\max(-A^{{({layer})}}, 0)){last_g} + b^{{({layer})}}"
    else:
        return f"A_{{+}}^{{({layer})}}{last_f} + A_{{-}}^{{({layer})}}{last_g} + b^{{({layer})}}"


def next_g(last_f: str, last_g: str, layer: int) -> str:
    if REPLACE_A_PLUS_AND_A_MINUS:
        return f"(-\max(-A^{{({layer})}}, 0)){last_f} + \max(A^{{({layer})}}, 0){last_g}"
    else:
        return f"A_{{-}}^{{({layer})}}{last_f} + A_{{+}}^{{({layer})}}{last_g}"


def next_f(last_g: str, last_h: str) -> str:
    return f"\max({last_h}, {last_g})"


def build_polynomial(n_layers: int) -> str:
    # initial h and g
    if REPLACE_A_PLUS_AND_A_MINUS:
        g = f"(-\max(-A^{{(0)}}, 0))x"
        h = f"\max(A^{{(0)}}, 0)x + b^{{(0)}}"
    else:
        g = f"A_{{-}}^{{(0)}}x"
        h = f"A_{{+}}^{{(0)}}x + b^{{(0)}}"
    # iterate over the layers
    for layer in range(1, n_layers):
        f = next_f(g, h)
        h = next_h(f, g, layer)
        g = next_g(f, g, layer)
    # add the last f
    f = next_f(g, h)

    return f"$${f} - {g}$$"


def main():
    md_str = "# Polynomial Terms"
    for n_layers in range(1, 4):
        md_str += f"\n\n## {n_layers} Layers\n\n"
        md_str += build_polynomial(n_layers)
    print(md_str)


if __name__ == "__main__":
    main()
