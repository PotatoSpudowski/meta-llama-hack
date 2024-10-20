import { ThemeOptions } from "@mui/material"

export const theme: ThemeOptions = {
    typography: {
        fontFamily: ["'Montserrat', sans-serif;"].join(","),
        allVariants: {
            color: "#778f8d",
        },
    },
    components: {
        MuiTypography: {
            styleOverrides: {
                subtitle2: {
                    fontSize: "0.8rem",
                },
            }
        },
        MuiIconButton: {
            styleOverrides: {
                root: {
                    color: "#bababa",
                },
            }
        },
        MuiTableContainer: {
            styleOverrides: {
                root: {
                    boxShadow: "none",
                },
            }
        },
        MuiTableRow: {
            styleOverrides: {
                root: {
                    "&.Mui-selected, &.Mui-selected:hover": {
                        backgroundColor: "#4dd47e2e",
                    },
                },
            }

        },
        MuiCard: {
            styleOverrides: {
                root: {
                    boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)",
                    borderRadius: "10px",
                }
            }
        },
        // Still considering it
        // MuiTableCell: {
        //   root: {
        //     padding: "8px",
        //   },
        // },
        MuiCheckbox: {
            styleOverrides: {
                colorSecondary: {
                    "&:hover": {
                        backgroundColor: "#2133310f",
                    },
                    "&$checked": {
                        color: "#213331",
                    },
                },
            }
            //TODO: Color for checked and hover
        },
        MuiSelect: {
            styleOverrides: {
                root: {
                    padding: "10.5px 14px",
                    fontSize: "0.875rem",
                },
            }
        },

        MuiPaper: {
            styleOverrides: {
                root: {
                    backgroundColor: "#ffffff",
                },
            }

        },
        MuiButton: {
            styleOverrides: {
                root: {
                    textTransform: "none",
                },
            }

        },
    },
    palette: {
        text: {
            primary: "#213331",
            secondary: "#778f8d",
        },
        primary: {
            main: "#5CCA85",
            contrastText: "#fff",
        },
        secondary: {
            main: "#E85D67",
            contrastText: "#fff",
        },
        divider: "#f4f4f4",
    },
}
