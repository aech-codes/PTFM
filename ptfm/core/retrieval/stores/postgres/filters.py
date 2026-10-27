"""
PTFM PostgreSQL Filter Translation

File:
    ptfm/core/retrieval/stores/postgres/filters.py

Description:
    Translation of ``RetrievalFilter`` into SQL.

Why this is a module of its own
-------------------------------
    This is the only place in the retrieval backend where a caller's value
    reaches a place the database parses, so it is the only place a filter
    injection could occur. Isolating it makes the boundary auditable: a reviewer
    checking whether values are bound rather than interpolated reads one file,
    and a test proving it reads the same file.

    Every value, without exception, becomes a bound parameter. No branch below
    interpolates a value into SQL text — not a filter value, not a metadata key,
    not a numeric bound. What is composed structurally is drawn only from the
    ``FilterField`` and ``FilterOperator`` enumerations, whose members are fixed
    in source and cannot originate from a query.

    ``metadata_key`` deserves separate mention because it is the one name a
    caller supplies rather than selects. It is bound as a value to a ``jsonb``
    key operator, never composed as an identifier. A key of ``"a' OR '1'='1"``
    therefore looks up a metadata key spelled exactly that way and finds
    nothing, which is the correct answer.

Agreement with the in-memory reference
--------------------------------------
    The in-memory stores define what a filter means; this module has to
    reproduce that meaning in SQL rather than choose a reasonable SQL
    equivalent. Three behaviours are easy to get subtly wrong and are therefore
    named here:

    An ordered comparison against a non-numeric value is NO MATCH, not an
    error. SQL has no total cast, so this relies on ``ptfm_to_number``, which
    yields NULL; the comparison becomes NULL and a ``WHERE`` clause treats NULL
    as not matching.

    A sequence-valued or multi-valued attribute is matched EXISTENTIALLY, and
    for negative operators too. A node on pages 1 and 2 satisfies ``PAGE_NUMBER
    NOT_EQUALS 1``, because page 2 differs from 1. This is surprising but it is
    what the reference does, so it is what this does.

    An attribute the node does not carry satisfies exactly the negative
    operators. A node with no pages, or without the metadata key, matches
    ``NOT_EQUALS`` and ``NOT_IN`` and fails everything else. Treating absence as
    a match for both polarities would make ``EQUALS`` and ``NOT_EQUALS``
    simultaneously true for it.

Dependency contract
-------------------
    Depends on ``psycopg.sql`` for composition and on the retrieval models for
    the filter vocabulary. It performs no I/O and holds no connection, so its
    entire behaviour is testable without a database — which is what allows the
    injection tests to be cheap enough to run on every change.
"""

from __future__ import annotations

from typing import Any, Final, Optional

from psycopg import sql

from ptfm.core.retrieval.exceptions import RetrievalConfigurationError
from ptfm.core.retrieval.models import (
    FilterField,
    FilterOperator,
    RetrievalFilter,
)

__all__ = [
    "SqlCondition",
    "FilterTranslator",
    "TRUE_CONDITION",
    "FALSE_CONDITION",
]


class SqlCondition:
    """
    A boolean SQL expression together with the parameters it binds.

    The pairing is the point. A fragment and its parameters separated even
    briefly can be recombined in the wrong order, and the resulting query is
    usually valid SQL asking a different question. Keeping them in one immutable
    object means a caller cannot hold one without the other.

    Attributes
    ----------
    expression:
        A composable SQL expression evaluating to boolean.
    params:
        Parameters bound by the expression, in the order its placeholders
        appear.
    """

    __slots__ = ("expression", "params")

    def __init__(
        self,
        expression: sql.Composable,
        params: tuple[Any, ...] = (),
    ) -> None:
        """
        Store an expression and its parameters.

        Parameters
        ----------
        expression:
            The SQL expression.
        params:
            The parameters it binds.
        """
        self.expression = expression
        self.params = params

    def __repr__(self) -> str:
        """
        Describe the condition for diagnostics.

        Returns
        -------
        str
            A representation naming the parameter count rather than the values,
            so that a log line cannot leak a filter value.
        """
        return f"SqlCondition(params={len(self.params)})"


#: The condition matching everything, for an empty filter set.
TRUE_CONDITION: Final[SqlCondition] = SqlCondition(sql.SQL("TRUE"))

#: The condition matching nothing, for a constraint that cannot be satisfied.
FALSE_CONDITION: Final[SqlCondition] = SqlCondition(sql.SQL("FALSE"))


#: Operators satisfied by an attribute the node does not carry.
_NEGATIVE: Final[frozenset[FilterOperator]] = frozenset(
    {FilterOperator.NOT_EQUALS, FilterOperator.NOT_IN}
)

#: Operators comparing magnitude rather than identity.
_ORDERED: Final[frozenset[FilterOperator]] = frozenset(
    {
        FilterOperator.GREATER_THAN_OR_EQUAL,
        FilterOperator.LESS_THAN_OR_EQUAL,
    }
)

#: Scalar node attributes, as text expressions.
#:
#: ORDINAL appears as a text cast rather than as its integer column because the
#: reference compares it as a string for identity operators: ordinal 5 does not
#: equal the filter value "05". The ordered operators use the integer column
#: directly, which is both correct and indexable.
_TEXT_COLUMN: Final[dict[FilterField, sql.Composable]] = {
    FilterField.DOCUMENT_ID: sql.Identifier("document_id"),
    FilterField.SOURCE_URI: sql.Identifier("source_uri"),
    FilterField.NODE_TYPE: sql.Identifier("node_type"),
    FilterField.ORDINAL: sql.SQL("{}::text").format(sql.Identifier("ordinal")),
}


class FilterTranslator:
    """
    Translate retrieval filters into SQL conditions.

    Stateless, and carries ``__slots__`` so it stays that way: an instance with
    no attributes cannot accumulate per-query state that would make two
    translations of the same filter differ.

    A caller may hold one instance for the process lifetime.
    """

    __slots__ = ()

    def translate_all(
        self,
        filters: tuple[RetrievalFilter, ...],
        *,
        column_prefix: Optional[str] = None,
    ) -> SqlCondition:
        """
        Translate a filter set into one conjunctive condition.

        Filters combine conjunctively, matching the reference: a node must
        satisfy all of them. Disjunction is expressed within a filter by ``IN``,
        which keeps the combinator single and its SQL unambiguous.

        Parameters
        ----------
        filters:
            The constraints. An empty set yields the condition matching
            everything, so a caller need not special-case the unfiltered query.
        column_prefix:
            Optional table alias qualifying every column reference, for a query
            joining the node table to another. Composed as an identifier, so it
            is quoted rather than interpolated.

        Returns
        -------
        SqlCondition
            The conjunction of the translated filters.

        Raises
        ------
        RetrievalConfigurationError
            If a field and operator combination has no translation.
        """
        if not filters:
            return TRUE_CONDITION

        translated = [
            self.translate(constraint, column_prefix=column_prefix)
            for constraint in filters
        ]

        return SqlCondition(
            sql.SQL(" AND ").join(
                sql.SQL("({})").format(condition.expression)
                for condition in translated
            ),
            tuple(
                parameter
                for condition in translated
                for parameter in condition.params
            ),
        )

    def translate(
        self,
        constraint: RetrievalFilter,
        *,
        column_prefix: Optional[str] = None,
    ) -> SqlCondition:
        """
        Translate a single filter into a SQL condition.

        Parameters
        ----------
        constraint:
            The constraint to translate.
        column_prefix:
            Optional table alias qualifying column references.

        Returns
        -------
        SqlCondition
            The condition, with every value bound as a parameter.

        Raises
        ------
        RetrievalConfigurationError
            If the field and operator combination has no translation.
        """
        column = _Column(column_prefix)

        if constraint.field is FilterField.HEADING_PATH:
            return self._translate_heading_path(constraint, column)
        if constraint.field is FilterField.PAGE_NUMBER:
            return self._translate_page_number(constraint, column)
        if constraint.field is FilterField.METADATA:
            return self._translate_metadata(constraint, column)
        return self._translate_scalar(constraint, column)

    def _translate_scalar(
        self,
        constraint: RetrievalFilter,
        column: _Column,
    ) -> SqlCondition:
        """
        Translate a filter over a single-valued node attribute.

        Parameters
        ----------
        constraint:
            The constraint to translate.
        column:
            Column reference builder.

        Returns
        -------
        SqlCondition
            The condition.

        Raises
        ------
        RetrievalConfigurationError
            If the field is not a scalar attribute.
        """
        expression = _TEXT_COLUMN.get(constraint.field)
        if expression is None:
            raise RetrievalConfigurationError(
                f"Field {constraint.field.value} has no scalar SQL "
                f"translation. Every FilterField member must be translated by "
                f"exactly one branch, so this indicates a field added to the "
                f"enumeration without a corresponding translation."
            )

        if constraint.operator in _ORDERED and (
            constraint.field is FilterField.ORDINAL
        ):
            # The integer column, not the text cast: an ordered comparison over
            # ordinals is numeric, and using the column directly keeps the
            # index usable.
            return _ordered(
                column.of(sql.Identifier("ordinal")),
                constraint,
                already_numeric=True,
            )

        qualified = column.qualify(expression, constraint.field)

        if constraint.operator in _ORDERED:
            return _ordered(qualified, constraint, already_numeric=False)
        return _identity(qualified, constraint)

    def _translate_heading_path(
        self,
        constraint: RetrievalFilter,
        column: _Column,
    ) -> SqlCondition:
        """
        Translate a filter over the heading path.

        ``STARTS_WITH`` compares a prefix slice, which is how a heading path
        selects a section together with everything beneath it. An array shorter
        than the requested prefix yields a shorter slice, which cannot equal the
        expected values, so the length check is implicit in the comparison
        rather than a separate clause that could disagree with it.

        Ordered operators match nothing: the reference declines to invent an
        ordering over heading paths, and a SQL array comparison would supply one
        that no other backend would reproduce.

        Parameters
        ----------
        constraint:
            The constraint to translate.
        column:
            Column reference builder.

        Returns
        -------
        SqlCondition
            The condition.
        """
        path = column.of(sql.Identifier("heading_path"))
        values = list(constraint.values)

        if constraint.operator in _ORDERED:
            return FALSE_CONDITION

        if constraint.operator is FilterOperator.STARTS_WITH:
            return SqlCondition(
                sql.SQL("{}[1:{}] = {}::text[]").format(
                    path, sql.Placeholder(), sql.Placeholder()
                ),
                (len(values), values),
            )
        if constraint.operator is FilterOperator.EQUALS:
            return SqlCondition(
                sql.SQL("{} = {}::text[]").format(path, sql.Placeholder()),
                (values,),
            )
        if constraint.operator is FilterOperator.NOT_EQUALS:
            return SqlCondition(
                sql.SQL("{} <> {}::text[]").format(path, sql.Placeholder()),
                (values,),
            )
        if constraint.operator is FilterOperator.IN:
            # Array overlap: some element of the path is among the values.
            return SqlCondition(
                sql.SQL("{} && {}::text[]").format(path, sql.Placeholder()),
                (values,),
            )
        return SqlCondition(
            sql.SQL("NOT ({} && {}::text[])").format(path, sql.Placeholder()),
            (values,),
        )

    def _translate_page_number(
        self,
        constraint: RetrievalFilter,
        column: _Column,
    ) -> SqlCondition:
        """
        Translate a filter over the pages a node originates on.

        Existential over the pages, for negative operators as well as positive
        ones, because that is what the reference does: a node on pages 1 and 2
        satisfies ``NOT_EQUALS 1`` by virtue of page 2. A node with no recorded
        pages satisfies exactly the negative operators.

        Parameters
        ----------
        constraint:
            The constraint to translate.
        column:
            Column reference builder.

        Returns
        -------
        SqlCondition
            The condition.
        """
        pages = column.of(sql.Identifier("page_numbers"))
        element = sql.SQL("page_element")

        if constraint.operator in _ORDERED:
            inner = _ordered(element, constraint, already_numeric=True)
        else:
            inner = _identity(
                sql.SQL("{}::text").format(element), constraint
            )

        absent_matches = sql.SQL(
            "TRUE" if constraint.operator in _NEGATIVE else "FALSE"
        )

        return SqlCondition(
            sql.SQL(
                "CASE WHEN cardinality({pages}) = 0 THEN {absent} ELSE EXISTS ("
                "SELECT 1 FROM unnest({pages}) AS {element} WHERE {inner}"
                ") END"
            ).format(
                pages=pages,
                absent=absent_matches,
                element=element,
                inner=inner.expression,
            ),
            inner.params,
        )

    def _translate_metadata(
        self,
        constraint: RetrievalFilter,
        column: _Column,
    ) -> SqlCondition:
        """
        Translate a filter over an enricher-supplied metadata value.

        The metadata key is bound as a parameter to the ``jsonb`` key operators,
        never composed as an identifier. This is the single place a caller
        supplies a name rather than selecting from an enumeration, so it is the
        single place where that distinction matters.

        Absence of the key satisfies exactly the negative operators, which is
        why the key existence test appears in the condition rather than being
        assumed.

        Parameters
        ----------
        constraint:
            The constraint to translate.
        column:
            Column reference builder.

        Returns
        -------
        SqlCondition
            The condition.
        """
        metadata = column.of(sql.Identifier("node_metadata"))
        key = constraint.metadata_key or ""

        present = sql.SQL("{} ? {}").format(metadata, sql.Placeholder())
        value = sql.SQL("{} ->> {}").format(metadata, sql.Placeholder())

        if constraint.operator in _ORDERED:
            inner = _ordered(value, constraint, already_numeric=False)
        else:
            inner = _identity(value, constraint)

        if constraint.operator in _NEGATIVE:
            return SqlCondition(
                sql.SQL("NOT ({present}) OR ({inner})").format(
                    present=present, inner=inner.expression
                ),
                (key, key) + inner.params,
            )

        return SqlCondition(
            sql.SQL("({present}) AND ({inner})").format(
                present=present, inner=inner.expression
            ),
            (key, key) + inner.params,
        )


class _Column:
    """
    Build optionally qualified column references.

    A table alias is composed as a quoted identifier rather than interpolated,
    so an alias containing a quote produces an invalid identifier rather than a
    query fragment.

    Attributes
    ----------
    prefix:
        The table alias, or ``None`` for unqualified references.
    """

    __slots__ = ("prefix",)

    def __init__(self, prefix: Optional[str]) -> None:
        """
        Store the alias.

        Parameters
        ----------
        prefix:
            The table alias, or ``None``.
        """
        self.prefix = prefix

    def of(self, identifier: sql.Composable) -> sql.Composable:
        """
        Qualify a bare column identifier.

        Parameters
        ----------
        identifier:
            The column identifier.

        Returns
        -------
        sql.Composable
            The qualified reference, or the identifier unchanged when no alias
            is in use.
        """
        if self.prefix is None:
            return identifier
        return sql.SQL("{}.{}").format(sql.Identifier(self.prefix), identifier)

    def qualify(
        self,
        expression: sql.Composable,
        field: FilterField,
    ) -> sql.Composable:
        """
        Qualify a possibly compound column expression.

        Rebuilds rather than wraps, because an alias cannot be prefixed onto an
        expression that already contains a cast: ``a.ordinal::text`` is correct
        and ``a.(ordinal::text)`` is not valid SQL.

        Parameters
        ----------
        expression:
            The unqualified expression, used when no alias is in use.
        field:
            The field the expression reads, used to rebuild it under an alias.

        Returns
        -------
        sql.Composable
            The qualified expression.
        """
        if self.prefix is None:
            return expression
        if field is FilterField.ORDINAL:
            return sql.SQL("{}::text").format(self.of(sql.Identifier("ordinal")))
        return self.of(_BARE_IDENTIFIER[field])


#: Bare column identifiers for the scalar fields, for alias qualification.
_BARE_IDENTIFIER: Final[dict[FilterField, sql.Identifier]] = {
    FilterField.DOCUMENT_ID: sql.Identifier("document_id"),
    FilterField.SOURCE_URI: sql.Identifier("source_uri"),
    FilterField.NODE_TYPE: sql.Identifier("node_type"),
    FilterField.ORDINAL: sql.Identifier("ordinal"),
}


def _identity(
    expression: sql.Composable,
    constraint: RetrievalFilter,
) -> SqlCondition:
    """
    Compare a text expression by identity or prefix.

    ``starts_with`` is used rather than ``LIKE`` so that a value containing
    ``%`` or ``_`` matches those characters literally. A ``LIKE`` translation
    would require escaping the value, and an escaping mistake is a silent
    change in meaning rather than an error.

    Parameters
    ----------
    expression:
        A SQL expression of text type.
    constraint:
        The constraint supplying the operator and values.

    Returns
    -------
    SqlCondition
        The condition.

    Raises
    ------
    RetrievalConfigurationError
        If the operator is not an identity or prefix comparison.
    """
    values = list(constraint.values)

    if constraint.operator is FilterOperator.EQUALS:
        return SqlCondition(
            sql.SQL("{} = {}").format(expression, sql.Placeholder()),
            (values[0],),
        )
    if constraint.operator is FilterOperator.NOT_EQUALS:
        return SqlCondition(
            sql.SQL("{} <> {}").format(expression, sql.Placeholder()),
            (values[0],),
        )
    if constraint.operator is FilterOperator.IN:
        return SqlCondition(
            sql.SQL("{} = ANY({}::text[])").format(
                expression, sql.Placeholder()
            ),
            (values,),
        )
    if constraint.operator is FilterOperator.NOT_IN:
        return SqlCondition(
            sql.SQL("NOT ({} = ANY({}::text[]))").format(
                expression, sql.Placeholder()
            ),
            (values,),
        )
    if constraint.operator is FilterOperator.STARTS_WITH:
        return SqlCondition(
            sql.SQL("starts_with({}, {})").format(expression, sql.Placeholder()),
            (values[0],),
        )

    raise RetrievalConfigurationError(
        f"Operator {constraint.operator.value} is not an identity comparison "
        f"and reached the identity translation, which indicates an operator "
        f"added to the enumeration without a translation."
    )


def _ordered(
    expression: sql.Composable,
    constraint: RetrievalFilter,
    *,
    already_numeric: bool,
) -> SqlCondition:
    """
    Compare an expression by magnitude.

    Both sides pass through ``ptfm_to_number`` unless already numeric, so a
    non-numeric operand yields NULL and the comparison does not match. That
    reproduces the reference, which treats an ordered comparison against a
    non-numeric value as no match rather than as an error.

    Parameters
    ----------
    expression:
        A SQL expression, numeric or text.
    constraint:
        The constraint supplying the operator and bound.
    already_numeric:
        Whether the expression is of numeric type, in which case it needs no
        conversion.

    Returns
    -------
    SqlCondition
        The condition.

    Raises
    ------
    RetrievalConfigurationError
        If the operator is not an ordered comparison.
    """
    if constraint.operator is FilterOperator.GREATER_THAN_OR_EQUAL:
        operator = sql.SQL(">=")
    elif constraint.operator is FilterOperator.LESS_THAN_OR_EQUAL:
        operator = sql.SQL("<=")
    else:
        raise RetrievalConfigurationError(
            f"Operator {constraint.operator.value} is not an ordered "
            f"comparison and reached the ordered translation, which indicates "
            f"an operator added to the enumeration without a translation."
        )

    left = (
        expression
        if already_numeric
        else sql.SQL("ptfm_to_number({})").format(expression)
    )

    return SqlCondition(
        sql.SQL("{left} {operator} ptfm_to_number({bound})").format(
            left=left, operator=operator, bound=sql.Placeholder()
        ),
        (constraint.values[0],),
    )
