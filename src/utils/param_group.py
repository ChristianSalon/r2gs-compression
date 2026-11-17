from argparse import ArgumentParser

from r2_gaussian.utils.argument_utils import ParamGroup


class ExtendedParamGroup(ParamGroup):
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)

        for key, value_original in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]

            t = type(value_original)
            value = value_original if not fill_none else None

            arg_flags = ["--" + key]
            if shorthand:
                arg_flags.append("-" + key[0:1])

            if t == bool:
                group.add_argument(*arg_flags, default=value, action="store_true")
            elif t == list:
                # Infer element type from the default list (defaults to str if empty)
                item_type = type(value_original[0]) if value_original else str
                group.add_argument(*arg_flags, default=value, nargs="+", type=item_type)
            else:
                group.add_argument(*arg_flags, default=value, type=t)
