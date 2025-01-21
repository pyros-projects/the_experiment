from fasthtml.common import *
from monsterui.all import *
from fasthtml.components import Sl_tab_group, Sl_tab, Sl_tab_panel, Sl_select, Sl_option, Sl_icon
from asteval import Interpreter # For safe evaluation

# Define allowed operations and their descriptions
NUMERIC_OPS = {
    'max': 'Maximum of values',
    'min': 'Minimum of values', 
    'sum': 'Sum values',
    'avg': 'Average of values',
    '+': 'Addition',
    '-': 'Subtraction',
    '*': 'Multiplication',
    '/': 'Division',
    '>': 'Greater than',
    '<': 'Less than',
    '>=': 'Greater than or equal',
    '<=': 'Less than or equal',
}

STRING_OPS = {
    '+': 'Concatenate strings',
    'SPLIT': 'Split string by delimiter: SPLIT(text, ",")',
    'JOIN': 'Join array with delimiter: JOIN(array, ",")',
    'REPLACE': 'Replace text: REPLACE(text, old, new)',
    'CONTAINS': 'Check if string contains: CONTAINS(text, search)',
    'STARTS_WITH': 'Check if starts with: STARTS_WITH(text, prefix)',
    'ENDS_WITH': 'Check if ends with: ENDS_WITH(text, suffix)',
    'UPPER': 'Convert to uppercase: UPPER(text)',
    'LOWER': 'Convert to lowercase: LOWER(text)',
    'TRIM': 'Remove whitespace: TRIM(text)',
    'LEN': 'Get length: LEN(text)',
    'SUBSTR': 'Get substring: SUBSTR(text, start, length)'
}

COMMON_OPS = {
    '==': 'Equal to',
    '!=': 'Not equal to',
    'and': 'Logical AND',
    'or': 'Logical OR',
    'if': 'Conditional (if/else)'
}

def create_string_functions():
    return {
        'SPLIT': lambda x, sep: x.split(sep),
        'JOIN': lambda arr, sep: sep.join(arr),
        'REPLACE': lambda x, old, new: x.replace(old, new),
        'CONTAINS': lambda x, search: search in x,
        'STARTS_WITH': lambda x, prefix: x.startswith(prefix),
        'ENDS_WITH': lambda x, suffix: x.endswith(suffix),
        'UPPER': lambda x: x.upper(),
        'LOWER': lambda x: x.lower(),
        'TRIM': lambda x: x.strip(),
        'LEN': len,
        'SUBSTR': lambda x, start, length: x[start:start+length]
    }

def create_rule_editor(rt):
    @rt("/validate_rule")
    def post(rule: str):
        try:
            if rule.startswith("input"):
                # Validate input definition
                parts = rule.split("=")
                if len(parts) != 2 or parts[1].strip() not in ['decimal', 'integer', 'boolean', 'string']:
                    return P("Input must be defined as decimal, integer, boolean, or string", 
                           cls="text-red-500")
                return P("Valid input definition", cls="text-green-500")
            
            if "=" not in rule:
                return P("Rule must contain assignment (=)", cls="text-red-500")
                
            left, expr = rule.split("=", 1)
            left = left.strip()
            
            if not (left.startswith("inter") or left == "output"):
                return P("Left side must start with 'inter' or be 'output'", 
                       cls="text-red-500")
            
            # Create safe interpreter with numeric and string functions
            aeval = Interpreter(
                use_numpy=False,
                minimal=True,
                functions={
                    **{"max": max, "min": min, "sum": sum, 
                       "avg": lambda x: sum(x)/len(x)},
                    **create_string_functions()
                }
            )
            
            # Test with both numeric and string values
            test_dict = {
                "input1": 1.0, 
                "input2": "test string",
                "inter1": 3.0,
                "inter2": "another string"
            }
            aeval.symtable.update(test_dict)
            aeval.eval(expr)
            
            return P("Valid rule", cls="text-green-500")
            
        except Exception as e:
            return P(f"Invalid rule: {str(e)}", cls="text-red-500")

    return Container(cls="p-8 space-y-6")(
        H2("Rule Editor"),
        P("Define input processing rules", cls=TextFont.muted_sm),
        
        # Operation reference tabs
        Sl_tab_group()(
            Sl_tab("Numeric Ops", slot="nav", panel="numeric", active=True),
            Sl_tab("String Ops", slot="nav", panel="string"),
            Sl_tab("Common Ops", slot="nav", panel="common"),
            
            Sl_tab_panel(
                Card(
                    Table(
                        Tr(Th("Operation"), Th("Description")),
                        *[Tr(Td(op), Td(desc)) for op, desc in NUMERIC_OPS.items()],
                        cls="text-sm"
                    )
                ),
                name="numeric"
            ),
            Sl_tab_panel(
                Card(
                    Table(
                        Tr(Th("Operation"), Th("Description")),
                        *[Tr(Td(op), Td(desc)) for op, desc in STRING_OPS.items()],
                        cls="text-sm"
                    )
                ),
                name="string"
            ),
            Sl_tab_panel(
                Card(
                    Table(
                        Tr(Th("Operation"), Th("Description")),
                        *[Tr(Td(op), Td(desc)) for op, desc in COMMON_OPS.items()],
                        cls="text-sm"
                    )
                ),
                name="common"
            )
        ),
        
        # Example rules
        Card(
            H3("Example Rules", slot="header"),
            Pre("""# Input definitions
input1=string
input2=decimal

# String processing
inter1 = UPPER(input1)
inter2 = SPLIT(inter1, ",")
inter3 = JOIN(inter2, "-")
inter4 = REPLACE(inter3, "old", "new")

# Mixed operations
output = inter4 + " value:" if input2 > 10 else LOWER(inter1)""",
                cls="text-sm text-slate-600 bg-slate-50 p-4 rounded"
            )
        ),
        
        # Rule editor
        Card(
            H3("Rules", slot="header"),
            Form(cls="space-y-4")(
                TextArea(
                    label="Input Definitions",
                    placeholder="input1=string\ninput2=decimal",
                    rows=2,
                    hx_post="/validate_rule",
                    hx_trigger="change",
                    hx_target="#input-validation"
                ),
                Div(id="input-validation"),
                
                TextArea(
                    label="Rules (one per line)",
                    placeholder="""inter1 = UPPER(input1)
inter2 = SPLIT(inter1, ",")
output = inter2 + str(input2) if CONTAINS(inter1, "test") else "default\"""",
                    rows=10,
                    hx_post="/validate_rule",
                    hx_trigger="change",
                    hx_target="#rule-validation"
                ),
                Div(id="rule-validation"),
                
                DivRAligned(
                    Button("Save Rules", cls=ButtonT.primary),
                    cls="pt-4"
                )
            )
        )
    )